from typing import Type
from .interfaces.model_client_interface import ModelClientInterface
from .interfaces.batch_model_interface import BatchModelInterface
from .interfaces.instant_model_interface import InstantModelInterface
from .model_client import ModelClient
import tiktoken


class ModelFactory:
    def __init__(self):
        self.models = {}

    def register_model(
        self,
        name: str,
        tokenizer,
        input_cost_per_million: float,
        output_cost_per_million: float,
        rpm_limit: int,
        rpd_limit: int,
        tpm_limit: int,
        tpd_limit: int,
        batch_queue_limit: int,
        batch_model_class: Type[BatchModelInterface],
        instant_model_class: Type[InstantModelInterface],
    ):
        self.models[name.lower()] = {
            "batch_model_class": batch_model_class,
            "instant_model_class": instant_model_class,
            "tokenizer": tokenizer,
            "input_cost_per_million": input_cost_per_million,
            "output_cost_per_million": output_cost_per_million,
            "rpm_limit": rpm_limit,
            "rpd_limit": rpd_limit,
            "tpm_limit": tpm_limit,
            "tpd_limit": tpd_limit,
            "batch_queue_limit": batch_queue_limit,
        }

    def get_model(self, name: str) -> ModelClientInterface:
        model_info = self.models.get(name.lower())
        if model_info is None:
            raise ValueError(f"Model '{name}' not found")

        tokenizer = tiktoken.get_encoding("cl100k_base")

        return ModelClient(
            model_name=name,
            tokenizer=tokenizer,
            input_cost_per_million=model_info["input_cost_per_million"],
            output_cost_per_million=model_info["output_cost_per_million"],
            batch_queue_limit=model_info["batch_queue_limit"],
            rpm_limit=model_info["rpm_limit"],
            rpd_limit=model_info["rpd_limit"],
            tpm_limit=model_info["tpm_limit"],
            tpd_limit=model_info["tpd_limit"],
            batch_model_class=model_info["batch_model_class"],
            instant_model_class=model_info["instant_model_class"],
        )

    def get_registered_models(self):
        return list(self.models.keys())
