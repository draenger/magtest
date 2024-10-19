from abc import ABC, abstractmethod
from ai_models.interfaces.model_client_interface import ModelClientInterface


class BenchmarkInterface(ABC):
    @abstractmethod
    def run_benchmark(self, model: ModelClientInterface, in_batch: bool = False):
        pass

    @abstractmethod
    def estimate_model_results(self, model: ModelClientInterface):
        pass

    @abstractmethod
    def check_and_process_batch_results(
        self,
        batch_id: str,
        model: ModelClientInterface,
        model_results: list,
        prepared_questions: dict,
    ) -> bool:
        pass
