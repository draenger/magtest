from typing import Callable
from .interfaces import BenchmarkInterface
from ai_models.interfaces.model_client_interface import ModelClientInterface


class BenchmarkRunner:
    def __init__(self, model_registry, benchmark_registry):
        self.model_factory = model_registry.get_factory()
        self.benchmark_factory = benchmark_registry.get_factory()

    def _run_benchmark_operation(
        self,
        operation_name: str,
        operation: Callable[[BenchmarkInterface, ModelClientInterface, bool], None],
        in_batch: bool = False,
    ):
        models = self.model_factory.get_registered_models()
        benchmarks = self.benchmark_factory.get_registered_benchmarks()

        for benchmark_name in benchmarks:
            try:
                benchmark = self.benchmark_factory.get_benchmark(benchmark_name)
            except Exception as e:
                print(
                    f"Error creating benchmark instance for {benchmark_name}: {str(e)}"
                )
                import traceback

                traceback.print_exc()
                continue

            for model_name in models:
                try:
                    model = self.model_factory.get_model(model_name)
                    operation(benchmark, model, in_batch)
                except Exception as e:
                    print(
                        f"Error running {operation_name} for benchmark {benchmark_name} and model {model_name}: {str(e)}"
                    )
                    import traceback

                    traceback.print_exc()

    def estimate_model_results(self):
        self._run_benchmark_operation(
            "estimate_model_results",
            lambda benchmark, model, _: benchmark.estimate_model_results(model),
        )

    def run_benchmarks(self, in_batch: bool = False):
        self._run_benchmark_operation(
            "run_benchmarks",
            lambda benchmark, model, in_batch: benchmark.run_benchmark(model, in_batch),
            in_batch,
        )

    def check_and_process_batch_results(self):
        self._run_benchmark_operation(
            "check_and_process_batch_results",
            lambda benchmark, model, _: benchmark.check_and_process_batch_results(
                model
            ),
        )
