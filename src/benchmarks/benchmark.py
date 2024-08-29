# benchmark.py
from abc import ABC, abstractmethod


class Benchmark(ABC):
    @abstractmethod
    def run_benchmark(self, model, max_tests_per_benchmark=1, num_few_shot=0):
        pass
