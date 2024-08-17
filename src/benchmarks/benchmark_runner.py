# benchmark_runner.py
class BenchmarkRunner:
    def __init__(self, model_factory, benchmark_factory):
        self.models = model_factory.get_registered_models()
        self.benchmarks = benchmark_factory.get_registered_benchmarks()
        self.model_factory = model_factory
        self.benchmark_factory = benchmark_factory

    def run_benchmarks(self):
        results = {}
        for model_name in self.models:
            model = self.model_factory.get_model(model_name)
            model_results = {}
            for benchmark_name in self.benchmarks:
                benchmark = self.benchmark_factory.get_benchmark(benchmark_name)
                score = benchmark.run_benchmark(model)
                model_results[benchmark_name] = score
            results[model_name] = model_results

        self.print_results(results)

    def print_results(self, results):
        for model_name, model_results in results.items():
            print(f"\nResults for {model_name}:")
            for benchmark_name, score in model_results.items():
                print(f"  {benchmark_name}: {score:.2f}")
