import uuid


class MMLUTestPreparation:
    def __init__(
        self, data_provider, prepared_question_repo, model_result_repo, test_session_id
    ):
        self.data_provider = data_provider
        self.prepared_question_repo = prepared_question_repo
        self.model_result_repo = model_result_repo
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

    def estimate_model_results(self, benchmark_name, model):
        existing_results = (
            self.model_result_repo.get_results_for_session_benchmark_and_model(
                self.test_session_id, benchmark_name, model.get_model_name()
            )
        )

        if existing_results:
            print(
                f"Estimation already exists for benchmark {benchmark_name}, session {self.test_session_id}, and model {model.get_model_name()}"
            )
            return

        prepared_questions = (
            self.prepared_question_repo.get_by_test_session_and_benchmark(
                self.test_session_id, benchmark_name
            )
        )

        for prepared_question in prepared_questions:
            estimated_in_tokens = model.estimate_tokens_amount(prepared_question.query)
            estimated_out_tokens = 1
            self.model_result_repo.add(
                prepared_question_id=prepared_question.id,
                model_name=model.get_model_name(),
                estimated_in_tokens=estimated_in_tokens,
                estimated_out_tokens=estimated_out_tokens,
                estimated_in_cost=estimated_in_tokens * model.get_model_in_token_cost(),
                estimated_out_cost=estimated_out_tokens
                * model.get_model_out_token_cost(),
            )
