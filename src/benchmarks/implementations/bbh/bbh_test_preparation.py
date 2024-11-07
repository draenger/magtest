import pandas as pd


class BBHTestPreparation:
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
        test_data = data["test"]
        icl_data = data["icl"]
        total_questions = len(test_data)

        print(
            f"Preparing {total_questions} tests for benchmark {benchmark_name}, session {self.test_session_id}, {num_few_shot}-shot"
        )

        # Group ICL data by category for few-shot examples
        icl_by_category = icl_data.groupby("category")

        for category, test_group in test_data.groupby("category"):
            # Get few-shot examples for this category
            category_icl_data = (
                icl_by_category.get_group(category)
                if category in icl_by_category.groups
                else pd.DataFrame()
            )

            # Get few-shot prompt for this category
            few_shot_prompt = (
                self.__get_few_shot_template__(
                    category_icl_data, num_few_shot, category
                )
                if num_few_shot > 0 and not category_icl_data.empty
                else ""
            )

            # Process each question in the category
            for _, row in test_group.iterrows():
                query = self.__prepare_query__(row, few_shot_prompt)
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

    def __prepare_query__(self, row, few_shot_prompt):
        """Prepare the query with the template"""
        main_question_template = """":\n Q: {question}\nA:Let's think step by step, and privide final answer at the end in format '#### ANSWER':"""
        return few_shot_prompt + main_question_template.format(**row)

    def __get_few_shot_template__(self, category_icl_data, num_few_shot, category):
        """Create few-shot template using examples from the same category"""
        few_shot_template = """Q: {question}\nA: {explanation}\n#### {answer}\n\n"""

        # Sample few-shot examples from the same category
        few_shot_examples = category_icl_data.sample(
            n=min(num_few_shot, len(category_icl_data))
        )

        # Get helper text from the first example of this category
        few_shot_helper_text = few_shot_examples.iloc[0]["helper_text"]

        few_shot_prompt = f"{few_shot_helper_text}\n\n" + "".join(
            few_shot_template.format(**example)
            for _, example in few_shot_examples.iterrows()
        )

        return few_shot_prompt

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
            estimated_out_tokens = 1000  # BBH requires longer responses
            self.model_result_repo.add(
                prepared_question_id=prepared_question.id,
                model_name=model.get_model_name(),
                estimated_in_tokens=estimated_in_tokens,
                estimated_out_tokens=estimated_out_tokens,
                estimated_in_cost=estimated_in_tokens * model.get_model_in_token_cost(),
                estimated_out_cost=estimated_out_tokens
                * model.get_model_out_token_cost(),
            )
