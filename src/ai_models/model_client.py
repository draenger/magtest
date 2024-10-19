from .interfaces.model_client_interface import ModelClientInterface
from .interfaces.batch_model_interface import BatchModelInterface
from .interfaces.instant_model_interface import InstantModelInterface


class ModelClient(ModelClientInterface):
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
        batch_model_class,
        instant_model_class,
    ):
        self._batch_model = batch_model_class(
            model_name,
            tokenizer,
            input_cost_per_million,
            output_cost_per_million,
            batch_queue_limit,
        )

        self._instant_model = instant_model_class(
            model_name,
            tokenizer,
            input_cost_per_million,
            output_cost_per_million,
            rpm_limit,
            rpd_limit,
            tpm_limit,
            tpd_limit,
        )

        self.model_name = model_name

    def get_batch_model(self) -> BatchModelInterface:
        return self._batch_model

    def get_instant_model(self) -> InstantModelInterface:
        return self._instant_model

    def get_model_name(self) -> str:
        return self.model_name
