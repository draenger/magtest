from abc import ABC, abstractmethod


class Benchmark(ABC):
    @abstractmethod
    def run_benchmark(self, model):
        pass

    @abstractmethod
    def estimate_model_results(self, model):
        pass
