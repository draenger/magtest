import json
from .implementations.mmlu import MMLUBenchmark, MMLUDataProvider, MMLUTestPreparation
from .implementations.gsm8k import (
    GSM8KBenchmark,
    GSM8KDataProvider,
    GSM8KTestPreparation,
)
from .implementations.bbh import (
    BBHBenchmark,
    BBHDataProvider,
    BBHTestPreparation,
)
from .benchmark_factory import BenchmarkFactory


class BenchmarkRegistry:
    def __init__(
        self,
        mmul_question_repository,
        gsm8k_question_repository,
        bbh_question_repository,
        prepared_question_repo,
        model_result_repo,
        batch_job_repo,
        test_session_id,
        benchmark_name_list=None,
    ):
        self.mmul_question_repository = mmul_question_repository
        self.gsm8k_question_repository = gsm8k_question_repository
        self.bbh_question_repository = bbh_question_repository
        self.prepared_question_repo = prepared_question_repo
        self.model_result_repo = model_result_repo
        self.batch_job_repo = batch_job_repo
        self.test_session_id = test_session_id
        self.config_file = "benchmark_config.json"
        self.benchmark_factory = BenchmarkFactory(benchmark_name_list)

    def load_config(self):
        """Load benchmark configuration from JSON file"""
        with open(self.config_file, "r") as file:
            return json.load(file)

    def register_benchmarks(self):
        """Register all available benchmarks based on configuration"""
        config = self.load_config()
        self._register_mmlu_benchmarks(config.get("mmlu", {}))
        self._register_gsm8k_benchmarks(config.get("gsm8k", {}))
        self._register_bbh_benchmarks(config.get("bbh", {}))

    def get_factory(self):
        return self.benchmark_factory

    def _register_mmlu_benchmarks(self, config):
        """Register MMLU benchmark variants based on configuration"""
        mmlu_test_preparation = self._create_mmlu_test_preparation()

        for variant in config.get("variants", []):
            self._register_benchmark(
                f"MMLU-{variant['shots']}Shot",
                MMLUBenchmark,
                mmlu_test_preparation,
                num_few_shot=variant["shots"],
                max_tokens=variant.get("max_tokens", 1),
                max_tests_per_category=variant.get("max_tests_per_category", 20),
            )

    def _register_gsm8k_benchmarks(self, config):
        """Register GSM8K benchmark variants based on configuration"""
        gsm8k_test_preparation = self._create_gsm8k_test_preparation()

        for variant in config.get("variants", []):
            self._register_benchmark(
                f"GSM8K-{variant['shots']}Shot",
                GSM8KBenchmark,
                gsm8k_test_preparation,
                num_few_shot=variant["shots"],
                max_tokens=variant.get("max_tokens", 500),
                max_tests_per_category=variant.get("max_tests_per_category", 50),
            )

    def _register_bbh_benchmarks(self, config):
        """Register BBH benchmark variants based on configuration"""
        bbh_test_preparation = self._create_bbh_test_preparation()

        for variant in config.get("variants", []):
            self._register_benchmark(
                f"BBH-{variant['shots']}Shot",
                BBHBenchmark,
                bbh_test_preparation,
                num_few_shot=variant["shots"],
                max_tokens=variant.get("max_tokens", 1000),
                max_tests_per_category=variant.get("max_tests_per_category", 30),
            )

    def _create_mmlu_test_preparation(self):
        """Create MMLU test preparation object"""
        mmlu_data_provider = MMLUDataProvider(self.mmul_question_repository)
        return MMLUTestPreparation(
            mmlu_data_provider,
            self.prepared_question_repo,
            self.model_result_repo,
            self.test_session_id,
        )

    def _create_gsm8k_test_preparation(self):
        """Create GSM8K test preparation object"""
        gsm8k_data_provider = GSM8KDataProvider(self.gsm8k_question_repository)
        return GSM8KTestPreparation(
            gsm8k_data_provider,
            self.prepared_question_repo,
            self.model_result_repo,
            self.test_session_id,
        )

    def _create_bbh_test_preparation(self):
        """Create BBH test preparation object"""
        bbh_data_provider = BBHDataProvider(self.bbh_question_repository)
        return BBHTestPreparation(
            bbh_data_provider,
            self.prepared_question_repo,
            self.model_result_repo,
            self.test_session_id,
        )

    def _register_benchmark(
        self,
        name: str,
        benchmark_class,
        test_preparation,
        num_few_shot: int,
        max_tokens: int,
        max_tests_per_category: int,
    ):
        """Register a single benchmark with specified parameters"""
        self.benchmark_factory.register_benchmark(
            name,
            benchmark_class,
            test_session_id=self.test_session_id,
            prepared_question_repo=self.prepared_question_repo,
            model_result_repo=self.model_result_repo,
            batch_job_repo=self.batch_job_repo,
            test_preparation=test_preparation,
            max_tests_per_category=max_tests_per_category,
            num_few_shot=num_few_shot,
            max_tokens=max_tokens,
        )

    def print_loaded_benchmarks(self):
        """Print information about loaded benchmarks"""
        loaded_benchmarks = self.benchmark_factory.get_registered_benchmarks()
        print("Loaded Benchmarks:")
        print("=================")
        for benchmark in loaded_benchmarks:
            print(f"- {benchmark}")
        print(f"\nTotal loaded benchmarks: {len(loaded_benchmarks)}")

        if self.benchmark_factory.benchmark_name_list:
            print(
                f"\nFiltered by benchmark list: {', '.join(self.benchmark_factory.benchmark_name_list)}"
            )
        else:
            print("\nNo benchmark list filter applied.")
