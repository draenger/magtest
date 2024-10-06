import os
import time
import openai
from dotenv import load_dotenv
from ...interfaces.ai_model_interface import AIModelInterface
from .open_ai_single_runner import OpenAISingleRunner
from .open_ai_batch_runner import OpenAIBatchRunner


class OpenAIModel(AIModelInterface):
    def __init__(
        self,
        model_name,
        tokenizer,
        input_cost_per_million,
        output_cost_per_million,
        rpm_limit,
    ):
        self.model_name = model_name
        self.tokenizer = tokenizer
        self.input_cost = input_cost_per_million / 1_000_000
        self.output_cost = output_cost_per_million / 1_000_000
        load_dotenv()
        self.single_runner = OpenAISingleRunner(self, rpm_limit)
        self.batch_runner = OpenAIBatchRunner(self)

    def get_model_name(self):
        return self.model_name

    def get_model_in_token_cost(self):
        return self.input_cost

    def get_model_out_token_cost(self):
        return self.output_cost

    def get_model_in_token_used(self):
        return self.single_runner.usage["prompt_tokens"]

    def get_model_out_token_used(self):
        return self.single_runner.usage["completion_tokens"]

    def predict(self, prompt):
        return self.single_runner.predict(prompt)

    def estimate_tokens_ammount(self, text):
        return len(self.tokenizer.encode(text))

    def add_batch_request(self, custom_id, model, messages, max_tokens=1):
        self.batch_runner.add_request(custom_id, model, messages, max_tokens=max_tokens)

    def run_batch(self, benchmark_name, metadata=None):
        return self.batch_runner.run_batch(benchmark_name, self.model_name, metadata)

    def check_batch_results(self, benchmark_name, batch_id):
        return self.batch_runner.check_for_results(
            benchmark_name, self.model_name, batch_id
        )
