import random
from .ai_model import AIModel


class TestModel(AIModel):
    def __init__(
        self,
        model_name="TestModel",
        tokenizer=None,
        input_cost_per_million=1,
        output_cost_per_million=1,
    ):
        self.model_name = model_name
        self.usage = {"prompt_tokens": 0, "completion_tokens": 0}
        self.tokenizer = tokenizer
        self.input_cost = input_cost_per_million / 1_000_000
        self.output_cost = output_cost_per_million / 1_000_000

    def get_model_name(self):
        return self.model_name

    def get_model_in_token_cost(self):
        return self.input_cost

    def get_model_out_token_cost(self):
        return self.output_cost

    def get_model_in_token_used(self):
        return self.usage["prompt_tokens"]

    def get_model_out_token_used(self):
        return self.usage["completion_tokens"]

    def predict(self, prompt):
        self.usage["prompt_tokens"] = len(self.tokenizer.encode(prompt))
        self.usage["completion_tokens"] = 1
        return random.choice(["A", "B", "C", "D"])

    def estimate_tokens_ammount(self, text):
        return len(self.tokenizer.encode(text))
