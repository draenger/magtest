from benchmarks.mmul import MMULDataProvider, MMULBenchmark, MMULTestPreparation
from benchmarks import BenchmarkFactory


class BenchmarkRegistry:
    def __init__(
        self,
        mmul_question_repository,
        prepared_question_repo,
        model_result_repo,
        test_session_id,
        max_tests_per_benchmark,
    ):
        self.mmul_question_repository = mmul_question_repository
        self.prepared_question_repo = prepared_question_repo
        self.model_result_repo = model_result_repo
        self.test_session_id = test_session_id
        self.max_tests_per_benchmark = max_tests_per_benchmark
        self.benchmark_factory = BenchmarkFactory()

    def register_mmul_benchmarks(self):
        mmul_data_provider = MMULDataProvider(self.mmul_question_repository)
        mmul_test_preparation = MMULTestPreparation(
            mmul_data_provider, self.prepared_question_repo, self.test_session_id
        )

        self.benchmark_factory.register_benchmark(
            "MMUL-0Shot",
            MMULBenchmark,
            test_session_id=self.test_session_id,
            prepared_question_repo=self.prepared_question_repo,
            model_result_repo=self.model_result_repo,
            test_preparation=mmul_test_preparation,
            max_tests_per_benchmark=self.max_tests_per_benchmark,
            num_few_shot=0,
        )

        self.benchmark_factory.register_benchmark(
            "MMUL-5Shot",
            MMULBenchmark,
            test_session_id=self.test_session_id,
            prepared_question_repo=self.prepared_question_repo,
            model_result_repo=self.model_result_repo,
            test_preparation=mmul_test_preparation,
            max_tests_per_benchmark=self.max_tests_per_benchmark,
            num_few_shot=5,
        )

    def get_factory(self):
        return self.benchmark_factory
