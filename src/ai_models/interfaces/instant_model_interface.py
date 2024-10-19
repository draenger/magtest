from abc import ABC, abstractmethod
from ..dto.usage import Usage
from ..dto.instant_response import InstantResponse


class InstantModelInterface(ABC):
    @abstractmethod
    def __init__(
        self,
        model_name: str,
        tokenizer,
        input_cost_per_million: float,
        output_cost_per_million: float,
        rpm_limit: int,
        rpd_limit: int,
        tpm_limit: int,
        tpd_limit: int,
    ):
        pass

    @abstractmethod
    def get_model_name(self) -> str:
        pass

    @abstractmethod
    def get_model_in_token_cost(self) -> float:
        pass

    @abstractmethod
    def get_model_out_token_cost(self) -> float:
        pass

    @abstractmethod
    def get_model_in_token_used(self) -> int:
        pass

    @abstractmethod
    def get_model_out_token_used(self) -> int:
        pass

    @abstractmethod
    def estimate_tokens_amount(self, text: str) -> int:
        pass

    @abstractmethod
    def wait_for_rate_limits(self):
        pass

    @abstractmethod
    def reset_usage(self):
        pass

    @abstractmethod
    def predict(self, prompt: str, max_tokens: int = 1) -> InstantResponse:
        pass

    @abstractmethod
    def update_usage(self, prompt_tokens: int, completion_tokens: int):
        pass
