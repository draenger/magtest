import uuid


class MMULTestPreparation:
    def __init__(self, data_provider, prepared_question_repo, test_session_id):
        self.data_provider = data_provider
        self.prepared_question_repo = prepared_question_repo
        self.test_session_id = test_session_id

    def prepare_test_data(self, benchmark_name, max_tests_per_benchmark, num_few_shot):
        existing_data = self.prepared_question_repo.get_by_test_session_and_benchmark(
            self.test_session_id, benchmark_name
        )

        if existing_data:
            print(
                f"Data already exists for benchmark {benchmark_name} and session {self.test_session_id}"
            )
            return

        data = self.data_provider.process_data(max_tests_per_benchmark)
        total_questions = len(data["test"])

        print(
            f"Preparing {total_questions} tests for benchmark {benchmark_name}, session {self.test_session_id}, {num_few_shot}-shot"
        )

        for category in data["test"]["category"].unique():
            category_test_data = data["test"][data["test"]["category"] == category]
            main_question_template = """Question: {question}\nOptions: A. {A}, B. {B}, C. {C}, D. {D}\nAnswer with just the letter of the correct option (A, B, C, or D)."""

            few_shot_prompt = (
                self.__get_few_shot_template__(data, category, num_few_shot)
                if num_few_shot > 0
                else ""
            )

            for _, row in category_test_data.iterrows():
                query = (
                    few_shot_prompt + main_question_template.format(**row)
                    if few_shot_prompt
                    else main_question_template.format(**row)
                )
                prepared_question = self.prepared_question_repo.add(
                    test_session_id=self.test_session_id,
                    benchmark_name=benchmark_name,
                    category=category,
                    query=query,
                    correct_answer=row["answer"],
                    num_few_shot=num_few_shot,
                )
                if prepared_question is None:
                    print(f"Error adding question for category {category}")

        print(
            f"Finished preparing {total_questions} tests for benchmark {benchmark_name}"
        )

    def __get_few_shot_template__(self, data, category, num_few_shot):
        category_dev_data = data["dev"][data["dev"]["category"] == category]
        few_shot_template = """Question: {question}\nOptions: A. {A}, B. {B}, C. {C}, D. {D}\nAnswer: {answer}"""
        few_shot_examples = category_dev_data.sample(
            n=min(num_few_shot, len(category_dev_data))
        )

        few_shot_prompt = "".join(
            few_shot_template.format(**example)
            for _, example in few_shot_examples.iterrows()
        )

        return few_shot_prompt + "\n"
