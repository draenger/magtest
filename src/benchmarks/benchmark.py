# benchmark.py
from abc import ABC, abstractmethod


class Benchmark(ABC):
    @abstractmethod
    def run_benchmark(self, model):
        pass
