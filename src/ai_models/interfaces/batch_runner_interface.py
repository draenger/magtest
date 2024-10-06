from abc import ABC, abstractmethod


class BatchRunnerInterface(ABC):
    @abstractmethod
    def __init__(self, client):
        pass

    @abstractmethod
    def add_request(self, custom_id, model, messages, max_tokens=1):
        pass

    @abstractmethod
    def run_batch(self, benchmark_name, model_name, metadata=None):
        pass

    @abstractmethod
    def check_for_results(self, benchmark_name, model_name, batch_id):
        pass
