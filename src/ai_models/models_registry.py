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
            input_cost_per_million=1,
            output_cost_per_million=1,
        )

    def register_test_openai_models(self):
        tokenizer = tiktoken.get_encoding("cl100k_base")

        models = [
            ("test-gpt-4o", 5, 15),
            ("test-gpt-4o-mini", 0.150, 0.600),
            ("test-gpt-4-turbo", 10, 30),
            ("test-gpt-4", 30, 60),
            ("test-gpt-3.5-turbo-0125", 0.50, 1.50),
        ]

        for model_name, input_cost, output_cost in models:
            self.model_factory.register_model(
                model_name,
                TestModel,
                tokenizer=tokenizer,
                input_cost_per_million=input_cost,
                output_cost_per_million=output_cost,
                model_name=model_name,
            )

    def register_openai_models(self):
        tokenizer = tiktoken.get_encoding("cl100k_base")
        rpm_limit = 500

        models = [
            ("gpt-4o", 5, 15, rpm_limit),
            ("gpt-4o-mini", 0.150, 0.600, rpm_limit),
            ("gpt-4-turbo", 10, 30, rpm_limit),
            ("gpt-4", 30, 60, rpm_limit),
            ("gpt-3.5-turbo-0125", 0.50, 1.50, rpm_limit),
        ]

        for model_name, input_cost, output_cost, rpm_limit in models:
            self.model_factory.register_model(
                model_name,
                OpenAIModel,
                tokenizer=tokenizer,
                input_cost_per_million=input_cost,
                output_cost_per_million=output_cost,
                model_name=model_name,
                rpm_limit=rpm_limit,
            )

    def get_factory(self):
        return self.model_factory
