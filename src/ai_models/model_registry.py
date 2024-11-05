import json
from .model_factory import ModelFactory
from .implementations.openai import OpenAIBatchModel, OpenAIInstantModel
from .implementations.anthropic import AnthropicBatchModel, AnthropicInstantModel
from .implementations.google import GoogleBatchModel, GoogleInstantModel
from .implementations.test import TestBatchModel, TestInstantModel


class ModelRegistry:
    def __init__(self, model_name_list=None):
        self.model_factory = ModelFactory()
        self.config_file = "model_config.json"
        self.model_name_list = model_name_list or []

    def load_config(self):
        with open(self.config_file, "r") as file:
            return json.load(file)

    def get_factory(self):
        return self.model_factory

    def register_models(self, model_type, batch_model_class, instant_model_class):
        config = self.load_config()
        for model in config.get("models", {}).get(model_type, []):
            if (
                not self.model_name_list
                or model["model_name"].lower() in self.model_name_list
            ):
                self.model_factory.register_model(
                    name=model["model_name"],
                    tokenizer="cl100k_base",
                    input_cost_per_million=model["input_cost_per_million"],
                    output_cost_per_million=model["output_cost_per_million"],
                    rpm_limit=model["rpm_limit"],
                    rpd_limit=model["rpd_limit"],
                    tpm_limit=model["tpm_limit"],
                    tpd_limit=model["tpd_limit"],
                    batch_queue_limit=model["batch_queue_limit"],
                    batch_model_class=batch_model_class,
                    instant_model_class=instant_model_class,
                )
            else:
                print(
                    f"Skipping model {model['model_name']} as it's not in the model_name_list"
                )

    def register_test_models(self):
        self.register_models("anthropic", TestBatchModel, TestInstantModel)
        self.register_models("openai", TestBatchModel, TestInstantModel)
        self.register_models("google", TestBatchModel, TestInstantModel)

    def register_production_models(self):
        self.register_models("anthropic", AnthropicBatchModel, AnthropicInstantModel)
        self.register_models("openai", OpenAIBatchModel, OpenAIInstantModel)
        self.register_models("google", GoogleBatchModel, GoogleInstantModel)

    def register_all_models(self):
        self.register_production_models()
        self.register_test_models()

    def print_loaded_models(self):
        loaded_models = self.model_factory.get_registered_models()
        print("Loaded Models:")
        print("==============")
        for model in loaded_models:
            print(f"- {model}")
        print(f"\nTotal loaded models: {len(loaded_models)}")

        if self.model_name_list:
            print(f"\nFiltered by model list: {', '.join(self.model_name_list)}")
        else:
            print("\nNo model list filter applied.")
