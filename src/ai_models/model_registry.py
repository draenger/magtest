import json
import tiktoken
from .model_factory import ModelFactory
from .implementations.openai.open_ai_model import OpenAIModel
from .implementations.anthropic.anthropic_model import AnthropicModel
from .implementations.test.test_model import TestModel


class ModelRegistry:
    def __init__(self):
        self.model_factory = ModelFactory()
        self.config_file = "model_config.json"

    def load_config(self):
        with open(self.config_file, "r") as file:
            return json.load(file)

    def get_factory(self):
        return self.model_factory

    def register_openai_models(self):
        config = self.load_config()
        tokenizer = tiktoken.get_encoding("cl100k_base")

        for model in config.get("models", {}).get("openai", []):
            self.model_factory.register_model(
                model["model_name"],
                OpenAIModel,
                tokenizer=tokenizer,
                input_cost_per_million=model["input_cost_per_million"],
                output_cost_per_million=model["output_cost_per_million"],
                rpm_limit=model["rpm_limit"],
                rpd_limit=model["rpd_limit"],
                tpm_limit=model["tpm_limit"],
                tpd_limit=model["tpd_limit"],
                batch_queue_limit=model["batch_queue_limit"],
            )

    def register_anthropic_models(self):
        config = self.load_config()
        tokenizer = tiktoken.get_encoding("cl100k_base")

        for model in config.get("models", {}).get("anthropic", []):
            self.model_factory.register_model(
                model["model_name"],
                AnthropicModel,
                tokenizer=tokenizer,
                input_cost_per_million=model["input_cost_per_million"],
                output_cost_per_million=model["output_cost_per_million"],
                rpm_limit=model["rpm_limit"],
                rpd_limit=model["rpd_limit"],
                tpm_limit=model["tpm_limit"],
                tpd_limit=model["tpd_limit"],
                batch_queue_limit=model["batch_queue_limit"],
            )

    def register_test_models(self):
        config = self.load_config()
        tokenizer = tiktoken.get_encoding("cl100k_base")

        for model_type in ["openai", "anthropic"]:
            for model in config.get("models", {}).get(model_type, []):
                test_model_name = f"test_{model['model_name']}"
                self.model_factory.register_model(
                    test_model_name,
                    TestModel,
                    tokenizer=tokenizer,
                    input_cost_per_million=model["input_cost_per_million"],
                    output_cost_per_million=model["output_cost_per_million"],
                    rpm_limit=model["rpm_limit"],
                    rpd_limit=model["rpd_limit"],
                    tpm_limit=model["tpm_limit"],
                    tpd_limit=model["tpd_limit"],
                    batch_queue_limit=model["batch_queue_limit"],
                )

    def register_all_models(self):
        self.register_openai_models()
        self.register_anthropic_models()
        self.register_test_models()
