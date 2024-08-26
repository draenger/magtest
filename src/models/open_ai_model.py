import os
import time
import openai
from dotenv import load_dotenv
from .ai_model import AIModel
from threading import Lock


class OpenAIModel(AIModel):
    def __init__(self, model_name, rpm_limit):
        self.model_name = model_name
        self.tokens_used = 0
        self.rpm_limit = rpm_limit
        self.request_times = []
        self.lock = Lock()
        load_dotenv()
        openai.api_key = os.getenv("OPENAI_API_KEY")

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

            self.tokens_used = 0
            response = openai.ChatCompletion.create(
                model=self.model_name,
                messages=[
                    # {
                    #     "role": "system",
                    #     "content": "Answer multiple choice questions. You are only allowed to answer A, B, C, or D.",
                    # },
                    {"role": "user", "content": prompt},
                ],
                max_tokens=1,
                n=1,
                stop=None,
                temperature=0.5,
            )

            self.tokens_used += response["usage"]["total_tokens"]
            answer = response.choices[0].message["content"].strip().upper()
            print(f"Answer: {answer}")
            if answer in ["A", "B", "C", "D"]:
                return answer
            else:
                return None
        except Exception as e:
            print(f"Error in prediction: {e}")
            return None

    def get_tokens_used(self):
        return self.tokens_used
