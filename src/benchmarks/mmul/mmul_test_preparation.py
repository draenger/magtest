import uuid


class MMULTestPreparation:
    def __init__(self, data_provider, prepared_question_repo):
        self.data_provider = data_provider
        self.prepared_question_repo = prepared_question_repo

    def prepare_test_data(self, max_tests_per_benchmark, num_few_shot):
        data = self.data_provider.process_data(max_tests_per_benchmark)
        test_session_id = str(uuid.uuid4())
        prepared_data = []

        few_shot_template = """Question: {question}
Options: A. {A}, B. {B}, C. {C}, D. {D}
Answer: {answer}

"""
        main_question_template = """Question: {question}
Options: A. {A}, B. {B}, C. {C}, D. {D}
Answer with just the letter of the correct option (A, B, C, or D)."""

        for category in data["test"]["category"].unique():
            category_test_data = data["test"][data["test"]["category"] == category]
            category_dev_data = data["dev"][data["dev"]["category"] == category]

            for _, row in category_test_data.iterrows():
                few_shot_examples = category_dev_data.sample(
                    n=min(num_few_shot, len(category_dev_data))
                )
                few_shot_prompt = "".join(
                    few_shot_template.format(**example)
                    for _, example in few_shot_examples.iterrows()
                )
                prompt = few_shot_prompt + main_question_template.format(**row)

                prepared_question = self.prepared_question_repo.add(
                    test_session_id=test_session_id,
                    benchmark_name="MMLU",
                    category=category,
                    query=prompt,
                    correct_answer=row["answer"],
                )

                prepared_data.append(
                    {
                        "id": prepared_question.id,
                        "prompt": prompt,
                        "answer": row["answer"],
                        "category": category,
                    }
                )

        return prepared_data, test_session_id
