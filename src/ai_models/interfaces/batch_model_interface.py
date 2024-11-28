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
    def run_batch(
        self,
        benchmark_name: str,
        metadata: Optional[dict] = None,
        test_session_id: int = None,
    ) -> List[str]:
        pass

    @abstractmethod
    def check_batch_status(self, batch_id: str) -> Optional[str]:
        pass

    @abstractmethod
    def process_batch_results(
        self, benchmark_name: str, batch_id: str, test_session_id: int
    ) -> Optional[BatchResponse]:
        pass

    @abstractmethod
    def retry_batch(
        self,
        batch_id: str,
        metadata: Optional[dict] = None,
    ) -> Optional[str]:
        pass

    @abstractmethod
    def cancel_batch(self, batch_id: str):
        pass

    @abstractmethod
    def list_batches(self, limit: int = 10):
        pass
