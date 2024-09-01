import uuid


class MMULTestPreparation:
    def __init__(self, data_provider, prepared_question_repo, test_session_id):
        self.data_provider = data_provider
        self.prepared_question_repo = prepared_question_repo
        self.test_session_id = test_session_id

    def prepare_test_data(self, max_tests_per_benchmark, num_few_shot):
        data = self.data_provider.process_data(max_tests_per_benchmark)

        for category in data["test"]["category"].unique():
            category_test_data = data["test"][data["test"]["category"] == category]
            main_question_template = """Question: {question}\nOptions: A. {A}, B. {B}, C. {C}, D. {D}\nAnswer with just the letter of the correct option (A, B, C, or D)."""

            few_shot_prompt = self.__get_few_shot_template__(
                data, category, num_few_shot
            )

            for _, row in category_test_data.iterrows():
                prepared_question = self.prepared_question_repo.add(
                    test_session_id=self.test_session_id,
                    benchmark_name="MMLU",
                    category=category,
                    query=few_shot_prompt + main_question_template.format(**row),
                    correct_answer=row["answer"],
                    num_few_shot=num_few_shot,
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
