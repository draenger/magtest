from abc import ABC, abstractmethod


class Benchmark(ABC):
    @abstractmethod
    def run_benchmark(self, model, in_batch=False):
        pass

    @abstractmethod
    def estimate_model_results(self, model):
        pass
