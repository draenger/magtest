class GSM8KTestPreparation:
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

        main_question_template = """Question: {question}\n Let's solve this step by step, with "####answer" in numeric format at the end:"""

        few_shot_prompt = (
            self.__get_few_shot_template__(data, num_few_shot)
            if num_few_shot > 0
            else ""
        )

        for _, row in data["test"].iterrows():
            query = (
                few_shot_prompt + main_question_template.format(**row)
                if few_shot_prompt
                else main_question_template.format(**row)
            )
            prepared_question = self.prepared_question_repo.add(
                test_session_id=self.test_session_id,
                benchmark_name=benchmark_name,
                category="math",
                query=query,
                correct_answer=row["answer"],
                num_few_shot=num_few_shot,
            )
            if prepared_question is None:
                print(f"Error adding question")

        print(
            f"Finished preparing {total_questions} tests for benchmark {benchmark_name}"
        )

    def __get_few_shot_template__(self, data, num_few_shot):
        few_shot_template = (
            """Question: {question}\nSolution: {full_solution}\nAnswer: {answer}\n\n"""
        )
        few_shot_examples = data["train"].sample(
            n=min(num_few_shot, len(data["train"]))
        )

        few_shot_prompt = "Here are some examples:\n\n" + "".join(
            few_shot_template.format(**example)
            for _, example in few_shot_examples.iterrows()
        )

        return few_shot_prompt + "\nNow let's solve the new question:\n\n"

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
            estimated_out_tokens = 500  # GSM8K wymaga dłuższych odpowiedzi
            self.model_result_repo.add(
                prepared_question_id=prepared_question.id,
                model_name=model.get_model_name(),
                estimated_in_tokens=estimated_in_tokens,
                estimated_out_tokens=estimated_out_tokens,
                estimated_in_cost=estimated_in_tokens * model.get_model_in_token_cost(),
                estimated_out_cost=estimated_out_tokens
                * model.get_model_out_token_cost(),
            )
