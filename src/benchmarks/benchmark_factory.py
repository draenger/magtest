class BenchmarkFactory:
    def __init__(self):
        self.benchmarks = {}

    def register_benchmark(
        self,
        name,
        benchmark_class,
        test_session_id,
        prepared_question_repo,
        model_result_repo,
        batch_job_repo,
        test_preparation,
        max_tests_per_benchmark,
        num_few_shot,
    ):
        self.benchmarks[name.lower()] = (
            benchmark_class,
            test_session_id,
            prepared_question_repo,
            model_result_repo,
            batch_job_repo,
            test_preparation,
            max_tests_per_benchmark,
            num_few_shot,
        )

    def get_benchmark(self, name):
        benchmark_info = self.benchmarks.get(name.lower())
        if benchmark_info is None:
            raise ValueError(f"Benchmark '{name}' not found")
        (
            benchmark_class,
            test_session_id,
            prepared_question_repo,
            model_result_repo,
            batch_job_repo,
            test_preparation,
            max_tests_per_benchmark,
            num_few_shot,
        ) = benchmark_info
        return benchmark_class(
            test_session_id,
            prepared_question_repo,
            model_result_repo,
            batch_job_repo,
            test_preparation,
            max_tests_per_benchmark,
            num_few_shot,
        )

    def get_registered_benchmarks(self):
        return list(self.benchmarks.keys())
