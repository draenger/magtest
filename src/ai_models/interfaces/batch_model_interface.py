from abc import ABC, abstractmethod
from typing import List, Optional
from ..dto.batch_response import BatchResponse


class BatchModelInterface(ABC):
    @abstractmethod
    def add_batch_request(
        self, custom_id: str, messages: List[dict], max_tokens: int = 1
    ):
        pass

    @abstractmethod
    def run_batch(self, benchmark_name: str, metadata: Optional[dict] = None) -> str:
        pass

    @abstractmethod
    def check_batch_results(
        self, benchmark_name: str, batch_id: str
    ) -> Optional[BatchResponse]:
        pass

    @abstractmethod
    def cancel_batch(self, batch_id: str):
        pass

    @abstractmethod
    def list_batches(self, limit: int = 10):
        pass

    @abstractmethod
    def process_batch_results(self, output_file_path: str) -> BatchResponse:
        pass
