from abc import ABC
from ...interfaces.benchmark_interface import BenchmarkInterface
from .mmul_one_by_one_runner import MMULOneByOneRunner
from .mmul_batch_runner import MMULBatchRunner
from ai_models.interfaces.model_client_interface import ModelClientInterface


class MMULBenchmark(BenchmarkInterface, ABC):

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
        self.benchmark_name = f"MMUL-{num_few_shot}Shot"

        self.test_preparation = test_preparation
        self.max_tests_per_benchmark = max_tests_per_benchmark
        self.num_few_shot = num_few_shot

        self.test_preparation.prepare_test_data(
            self.benchmark_name, self.max_tests_per_benchmark, self.num_few_shot
        )
        self.one_by_one_runner = MMULOneByOneRunner(model_result_repo)
        self.batch_runner = MMULBatchRunner(model_result_repo, batch_job_repo)

    def estimate_model_results(self, model: ModelClientInterface):
        self.test_preparation.estimate_model_results(
            self.benchmark_name, model.get_instant_model()
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
            )
        else:
            self.one_by_one_runner.run_benchmark_one_by_one(
                prepared_questions, model_results, model.get_instant_model()
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
