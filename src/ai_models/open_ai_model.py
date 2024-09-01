import os
import time
import openai
from dotenv import load_dotenv
from threading import Lock
from .ai_model import AIModel


class OpenAIModel(AIModel):
    def __init__(
        self,
        model_name,
        rpm_limit,
        tokenizer,
        input_cost_per_million,
        output_cost_per_million,
    ):
        self.model_name = model_name
        self.usage = {"prompt_tokens": 0, "completion_tokens": 0}
        self.rpm_limit = rpm_limit
        self.request_times = []
        self.lock = Lock()
        self.tokenizer = tokenizer
        load_dotenv()
        openai.api_key = os.getenv("OPENAI_API_KEY")
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

    def _wait_for_rate_limit(self):
        with self.lock:
            current_time = time.time()
            self.request_times = [
                t for t in self.request_times if current_time - t < 60
            ]

            if len(self.request_times) >= self.rpm_limit:
                sleep_time = 60 - (current_time - self.request_times[0])
                if sleep_time > 0:
                    time.sleep(sleep_time)

            self.request_times.append(time.time())

    def predict(self, prompt):
        try:
            self._wait_for_rate_limit()

            response = openai.ChatCompletion.create(
                model=self.model_name,
                messages=[
                    {"role": "user", "content": prompt},
                ],
                max_tokens=1,
                n=1,
                stop=None,
                temperature=0.5,
            )

            self.usage = response["usage"]

            return response.choices[0].message["content"].strip().upper()
        except Exception as e:
            print(f"Error in prediction: {e}")
            return None

    def estimate_tokens_ammount(self, text):
        return len(self.tokenizer.encode(text))
