from abc import ABC, abstractmethod
from .batch_model_interface import BatchModelInterface
from .instant_model_interface import InstantModelInterface


class ModelClientInterface(ABC):
    @abstractmethod
    def __init__(
        self,
        model_name: str,
        tokenizer,
        input_cost_per_million: float,
        output_cost_per_million: float,
        batch_queue_limit: int,
        rpm_limit: int,
        rpd_limit: int,
        tpm_limit: int,
        tpd_limit: int,
    ):
        pass

    @abstractmethod
    def get_batch_model(self) -> BatchModelInterface:
        pass

    @abstractmethod
    def get_instant_model(self) -> InstantModelInterface:
        pass
