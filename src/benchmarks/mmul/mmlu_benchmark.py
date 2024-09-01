from benchmarks import Benchmark
from time import perf_counter
import tiktoken


class MMULBenchmark(Benchmark):
    def __init__(
        self,
        test_session_id,
        prepared_question_repo,
        model_result_repo,
        test_preparation,
        max_tests_per_benchmark,
        num_few_shot,
    ):
        self.test_session_id = test_session_id
        self.prepared_question_repo = prepared_question_repo
        self.model_result_repo = model_result_repo
        self.test_preparation = test_preparation
        self.test_preparation.prepare_test_data(max_tests_per_benchmark, num_few_shot)

    def estimate_model_results(self, model):
        prepared_questions = self.prepared_question_repo.get_by_test_session(
            self.test_session_id
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

    def run_benchmark(self, model):
        model_results = self.model_result_repo.get_results_for_session_and_model(
            self.test_session_id, model.get_model_name()
        )
        prepared_questions = {
            q.id: q
            for q in self.prepared_question_repo.get_by_test_session(
                self.test_session_id
            )
        }

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

            correct_answer = prepared_question.correct_answer
            score = 1 if model_answer == correct_answer else 0

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
                score=score,
            )
