# benchmark_runner.py
class BenchmarkRunner:
    def __init__(self, model_factory, benchmark_factory):
        self.models = model_factory.get_registered_models()
        self.benchmarks = benchmark_factory.get_registered_benchmarks()
        self.model_factory = model_factory
        self.benchmark_factory = benchmark_factory

    def run_benchmarks(self):
        results = {benchmark_name: {} for benchmark_name in self.benchmarks}
        print("Running benchmarks...")

        for benchmark_name in self.benchmarks:
            print("\n=========================================")
            print(f"Running {benchmark_name} benchmark for all models...")
            benchmark = self.benchmark_factory.get_benchmark(benchmark_name)
            print()

            for model_name in self.models:
                model = self.model_factory.get_model(model_name)
                score = benchmark.run_benchmark(model)
                results[benchmark_name][model_name] = score
                print(f"Model: {model_name} score: {score}")

            print(f"\nFinished {benchmark_name} benchmark for all models")
        print("=========================================")
        print("\nAll benchmarks completed.")

        print(results)
