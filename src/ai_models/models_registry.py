import tiktoken
from ai_models import ModelFactory, OpenAIModel, TestModel


class ModelsRegistry:
    def __init__(self):
        self.model_factory = ModelFactory()

    def register_test_model(self):
        self.model_factory.register_model(
            "test",
            TestModel,
            tokenizer=tiktoken.get_encoding("cl100k_base"),
            input_cost_per_million=1,  # Example cost
            output_cost_per_million=2,  # Example cost
        )

    def register_openai_models(self):
        tokenizer = tiktoken.get_encoding("cl100k_base")

        self.model_factory.register_model(
            "gpt-4o",
            OpenAIModel,
            tokenizer=tokenizer,
            input_cost_per_million=5,
            output_cost_per_million=15,
            model_name="gpt-4o",
            rpm_limit=500,
        )
        self.model_factory.register_model(
            "gpt-4o-mini",
            OpenAIModel,
            tokenizer=tokenizer,
            input_cost_per_million=0.150,
            output_cost_per_million=0.600,
            model_name="gpt-4o-mini",
            rpm_limit=500,
        )
        self.model_factory.register_model(
            "gpt-4-turbo",
            OpenAIModel,
            tokenizer=tokenizer,
            input_cost_per_million=10,
            output_cost_per_million=30,
            model_name="gpt-4-turbo",
            rpm_limit=500,
        )
        self.model_factory.register_model(
            "gpt-4",
            OpenAIModel,
            tokenizer=tokenizer,
            input_cost_per_million=30,
            output_cost_per_million=60,
            model_name="gpt-4",
            rpm_limit=500,
        )
        self.model_factory.register_model(
            "gpt-3.5-turbo-0125",
            OpenAIModel,
            tokenizer=tokenizer,
            input_cost_per_million=0.50,
            output_cost_per_million=1.50,
            model_name="gpt-3.5-turbo-0125",
            rpm_limit=500,
        )

    def get_factory(self):
        return self.model_factory
