from typing import Type
from .interfaces import BenchmarkInterface


class BenchmarkFactory:
    def __init__(self, benchmark_name_list=None):
        self.benchmarks = {}
        self.benchmark_name_list = benchmark_name_list or []

    def register_benchmark(
        self,
        name: str,
        benchmark_class: Type[BenchmarkInterface],
        test_session_id,
        prepared_question_repo,
        model_result_repo,
        batch_job_repo,
        test_preparation,
        max_tests_per_benchmark,
        num_few_shot,
    ):
        if not self.benchmark_name_list or name.lower() in [
            x.lower() for x in self.benchmark_name_list
        ]:
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
        else:
            print(f"Skipping benchmark {name} as it's not in the benchmark_name_list")

    def get_benchmark(self, name) -> BenchmarkInterface:
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
