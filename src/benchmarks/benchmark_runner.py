import logging


class BenchmarkRunner:
    def __init__(self, models_registry, benchmark_registry):
        self.model_factory = models_registry.get_factory()
        self.benchmark_factory = benchmark_registry.get_factory()
        self.logger = logging.getLogger(__name__)

    def _run_benchmark_operation(self, operation_name, operation):
        models = self.model_factory.get_registered_models()
        benchmarks = self.benchmark_factory.get_registered_benchmarks()

        for benchmark_name in benchmarks:
            benchmark = self.benchmark_factory.get_benchmark(benchmark_name)

            for model_name in models:
                try:
                    model = self.model_factory.get_model(model_name)
                    operation(benchmark, model)
                except Exception as e:
                    self.logger.error(
                        f"Error running {operation_name} for benchmark {benchmark_name} and model {model_name}: {str(e)}",
                        exc_info=True,
                    )

    def estimate_model_results(self):
        self._run_benchmark_operation(
            "estimate_model_results",
            lambda benchmark, model: benchmark.estimate_model_results(model),
        )

    def run_benchmarks(self):
        self._run_benchmark_operation(
            "run_benchmarks",
            lambda benchmark, model: benchmark.run_benchmark(model),
        )
