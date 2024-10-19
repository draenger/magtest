from .interfaces.batch_model_interface import BatchModelInterface
from typing import List, Optional


class BaseBatchModel(BatchModelInterface):
    def __init__(
        self,
        model_name: str,
        tokenizer,
        input_cost_per_million: float,
        output_cost_per_million: float,
        batch_queue_limit: int,
    ):
        self.model_name = model_name
        self.tokenizer = tokenizer
        self.input_token_cost = input_cost_per_million / 1_000_000 / 2
        self.output_token_cost = output_cost_per_million / 1_000_000 / 2
        self.batch_queue_limit = batch_queue_limit
        self.requests = []

    def get_model_name(self) -> str:
        return self.model_name

    def get_model_in_token_cost(self) -> float:
        return self.input_token_cost

    def get_model_out_token_cost(self) -> float:
        return self.output_token_cost

    def estimate_tokens_amount(self, text: str) -> int:
        return len(self.tokenizer.encode(text))

    def add_batch_request(
        self, custom_id: str, messages: List[dict], max_tokens: int = 1
    ):
        raise NotImplementedError(
            "This method should be implemented in the child class"
        )

    def run_batch(
        self,
        benchmark_name: str,
        metadata: Optional[dict] = None,
        test_session_id: int = None,
    ) -> str:
        raise NotImplementedError(
            "This method should be implemented in the child class"
        )

    def check_batch_results(
        self, benchmark_name: str, batch_id: str, test_session_id: int
    ):
        raise NotImplementedError(
            "This method should be implemented in the child class"
        )

    def cancel_batch(self, batch_id: str):
        raise NotImplementedError(
            "This method should be implemented in the child class"
        )

    def list_batches(self, limit: int = 10):
        raise NotImplementedError(
            "This method should be implemented in the child class"
        )
