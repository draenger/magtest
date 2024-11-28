from abc import ABC
from ...interfaces.benchmark_interface import BenchmarkInterface
from .mmlu_one_by_one_runner import MMLUOneByOneRunner
from .mmlu_batch_runner import MMLUBatchRunner
from ai_models.interfaces.model_client_interface import ModelClientInterface


class MMLUBenchmark(BenchmarkInterface):

    def __init__(
        self,
        test_session_id,
        prepared_question_repo,
        model_result_repo,
        batch_job_repo,
        test_preparation,
        max_tests_per_category,
        num_few_shot,
        max_tokens,
    ):
        self.test_session_id = test_session_id
        self.prepared_question_repo = prepared_question_repo
        self.model_result_repo = model_result_repo
        self.batch_job_repo = batch_job_repo
        self.benchmark_name = f"MMLU-{num_few_shot}Shot"

        self.max_tokens = max_tokens
        self.max_tests_per_category = max_tests_per_category
        self.num_few_shot = num_few_shot

        self.test_preparation = test_preparation
        self.test_preparation.prepare_test_data(
            self.benchmark_name,
            self.max_tests_per_category,
            self.num_few_shot,
            self.max_tokens,
        )

        self.one_by_one_runner = MMLUOneByOneRunner(model_result_repo)
        self.batch_runner = MMLUBatchRunner(model_result_repo, batch_job_repo)

    def estimate_model_results(self, model: ModelClientInterface):
        self.test_preparation.estimate_model_results(
            self.benchmark_name, model.get_instant_model(), self.max_tokens
        )

    def run_benchmark(self, model: ModelClientInterface, in_batch: bool = False):
        model_results, prepared_questions = self.get_question_data(
            self.test_session_id, self.benchmark_name, model.get_model_name()
        )

        if in_batch:
            self.batch_runner.run_benchmark_batch(
                prepared_questions,
                model_results,
                model.get_batch_model(),
                self.test_session_id,
                self.benchmark_name,
                self.max_tokens,
            )
        else:
            self.one_by_one_runner.run_benchmark_one_by_one(
                prepared_questions,
                model_results,
                model.get_instant_model(),
                self.max_tokens,
            )

    def check_and_process_batch_results(
        self,
        model: ModelClientInterface,
    ) -> bool:
        model_results, prepared_questions = self.get_question_data(
            self.test_session_id, self.benchmark_name, model.get_model_name()
        )

        batches = self.batch_job_repo.get_by_test_session_and_benchmark_and_model(
            self.test_session_id, self.benchmark_name, model.get_model_name()
        )

        for batch in batches:
            # Skip batches that are already in retry status
            if batch.status == "retry":
                continue

            self.batch_runner.check_and_process_batch_results(
                batch.batch_id,
                model.get_batch_model(),
                model_results,
                prepared_questions,
                self.benchmark_name,
                self.test_session_id,
            )

        return True

    def get_question_data(
        self,
        test_session_id: str,
        benchmark_name: str,
        model_name: str,
    ):
        model_results = (
            self.model_result_repo.get_results_for_session_benchmark_and_model(
                test_session_id, benchmark_name, model_name
            )
        )

        prepared_questions = {
            q.id: q
            for q in self.prepared_question_repo.get_by_test_session_and_benchmark(
                test_session_id, benchmark_name
            )
        }

        return model_results, prepared_questions
