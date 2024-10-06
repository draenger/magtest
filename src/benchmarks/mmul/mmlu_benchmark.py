from benchmarks import Benchmark
from time import perf_counter
import json


class MMULBenchmark(Benchmark):

    def __init__(
        self,
        test_session_id,
        prepared_question_repo,
        model_result_repo,
        batch_job_repo,
        test_preparation,
        max_tests_per_benchmark,
        num_few_shot,
    ):
        self.test_session_id = test_session_id
        self.prepared_question_repo = prepared_question_repo
        self.model_result_repo = model_result_repo
        self.batch_job_repo = batch_job_repo
        self.test_preparation = test_preparation
        self.benchmark_name = "MMUL-" + str(num_few_shot) + "Shot"
        self.test_preparation.prepare_test_data(
            self.benchmark_name, max_tests_per_benchmark, num_few_shot
        )

    def estimate_model_results(self, model):
        existing_results = (
            self.model_result_repo.get_results_for_session_benchmark_and_model(
                self.test_session_id, self.benchmark_name, model.get_model_name()
            )
        )

        if existing_results:
            print(
                f"Estimation already exists for benchmark {self.benchmark_name}, session {self.test_session_id}, and model {model.get_model_name()}"
            )
            return

        prepared_questions = (
            self.prepared_question_repo.get_by_test_session_and_benchmark(
                self.test_session_id, self.benchmark_name
            )
        )

        for prepared_question in prepared_questions:
            estimated_in_tokens = model.estimate_tokens_ammount(prepared_question.query)
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

    def run_benchmark(self, model, in_batch=False):
        model_results = (
            self.model_result_repo.get_results_for_session_benchmark_and_model(
                self.test_session_id, self.benchmark_name, model.get_model_name()
            )
        )

        prepared_questions = {
            q.id: q
            for q in self.prepared_question_repo.get_by_test_session_and_benchmark(
                self.test_session_id, self.benchmark_name
            )
        }

        if in_batch:
            self.run_benchmark_batch(prepared_questions, model_results, model)
        else:
            self.run_benchmark_one_by_one(prepared_questions, model_results, model)

    def run_benchmark_one_by_one(self, prepared_questions, model_results, model):
        for model_result in model_results:
            prepared_question = prepared_questions.get(
                model_result.prepared_question_id
            )

            if not prepared_question:
                print(
                    f"Warning: PreparedQuestion not found for id {model_result.prepared_question_id}"
                )
                continue

            start_time = perf_counter()
            model_answer = model.predict(prepared_question.query)
            end_time = perf_counter()

            self.model_result_repo.update_execution_results(
                model_result.id,
                response=model_answer,
                actual_in_tokens=model.get_model_in_token_used(),
                actual_out_tokens=model.get_model_out_token_used(),
                actual_in_cost=model.get_model_in_token_used()
                * model.get_model_in_token_cost(),
                actual_out_cost=model.get_model_out_token_used()
                * model.get_model_out_token_cost(),
                execution_time=end_time - start_time,
                score=1 if model_answer == prepared_question.correct_answer else 0,
            )

    def run_benchmark_batch(self, prepared_questions, model_results, model):
        model_name = model.get_model_name()

        for model_result in model_results:
            prepared_question = prepared_questions.get(
                model_result.prepared_question_id
            )

            if not prepared_question:
                print(
                    f"Warning: PreparedQuestion not found for id {model_result.prepared_question_id}"
                )
                continue

            model.add_batch_request(
                custom_id=prepared_question.id,
                model=model_name,
                messages=[{"role": "user", "content": prepared_question.query}],
                max_tokens=1,
            )

        batch_id = model.run_batch(
            self.benchmark_name,
            metadata={
                "description": f"Benchmark[{self.benchmark_name}], Session[{self.test_session_id}], model[{model_name}]"
            },
        )

        self.batch_job_repo.add(
            test_session_id=self.test_session_id,
            benchmark_name=self.benchmark_name,
            model_name=model_name,
            batch_id=batch_id,
        )

        print(
            f"Batch - Benchmark[{self.benchmark_name}], Session[{self.test_session_id}], model[{model_name}] created with ID: {batch_id}"
        )

    def check_and_process_batch_results(
        self, batch_id, model, model_results, prepared_questions
    ):
        results_file = model.check_batch_results(self.benchmark_name, batch_id)
        if results_file:
            self.process_batch_results(
                results_file, model_results, prepared_questions, model
            )
            self.batch_job_repo.update_status(batch_id, "completed")
            return True
        else:
            print("Batch job not completed yet.")
            return False

    def process_batch_results(
        self, results_file, model_results, prepared_questions, model
    ):
        with open(results_file, "r") as f:
            results = json.load(f)

        for result in results:
            custom_id = result["custom_id"]
            model_result = next(
                (mr for mr in model_results if mr.prepared_question_id == custom_id),
                None,
            )
            prepared_question = prepared_questions.get(custom_id)

            if model_result and prepared_question:
                model_answer = result["choices"][0]["message"]["content"]
                self.model_result_repo.update_execution_results(
                    model_result.id,
                    response=model_answer,
                    actual_in_tokens=len(
                        prepared_question.query
                    ),  # This is an approximation
                    actual_out_tokens=1,
                    actual_in_cost=len(prepared_question.query)
                    * model.get_model_in_token_cost(),
                    actual_out_cost=model.get_model_out_token_cost(),
                    execution_time=0,  # We don't have this information for batch processing
                    score=1 if model_answer == prepared_question.correct_answer else 0,
                )
            else:
                print(f"Warning: Could not process result for custom_id {custom_id}")
