# benchmark_factory.py
class BenchmarkFactory:
    def __init__(self):
        self.benchmarks = {}

    def register_benchmark(self, name, benchmark_class, *args, **kwargs):
        self.benchmarks[name.lower()] = (benchmark_class, args, kwargs)

    def get_benchmark(self, name):
        benchmark_info = self.benchmarks.get(name.lower())
        if benchmark_info is None:
            raise ValueError(f"Benchmark '{name}' not found")
        benchmark_class, args, kwargs = benchmark_info
        return benchmark_class(*args, **kwargs)

    def get_registered_benchmarks(self):
        return list(self.benchmarks.keys())
