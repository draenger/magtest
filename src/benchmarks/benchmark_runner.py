# benchmark_runner.py
class BenchmarkRunner:
    def __init__(self, model_factory, benchmark_factory):
        self.model_factory = model_factory
        self.benchmark_factory = benchmark_factory

    def run_benchmarks(self, max_tests_per_benchmark=1, num_few_shot=0):
        models = self.model_factory.get_registered_models()
        benchmarks = self.benchmark_factory.get_registered_benchmarks()
        results = {benchmark_name: {} for benchmark_name in benchmarks}
        print("Running benchmarks...")

        for benchmark_name in benchmarks:
            print(f"\n{'=' * 40}")
            print(f"Running {benchmark_name} benchmark for all models...")
            benchmark = self.benchmark_factory.get_benchmark(benchmark_name)

            for model_name in models:
                try:
                    model = self.model_factory.get_model(model_name)
                    score = benchmark.run_benchmark(
                        model, max_tests_per_benchmark, num_few_shot
                    )
                    results[benchmark_name][model_name] = score
                    print(f"Model: {model_name} score: {score}")
                except Exception as e:
                    print(f"Error running {benchmark_name} for model {model_name}: {e}")

            print(f"\nFinished {benchmark_name} benchmark for all models")
        print(f"{'=' * 40}")
        print("\nAll benchmarks completed.")

        return results
