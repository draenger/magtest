import time
import openai
from threading import Lock
from ...interfaces import SingleRunnerInterface


class OpenAISingleRunner(SingleRunnerInterface):
    def __init__(self, client, rpm_limit, tpm_limit=None, tpd_limit=None):
        self.client = client
        self.rpm_limit = rpm_limit
        self.request_times = []
        self.lock = Lock()
        self.usage = {"prompt_tokens": 0, "completion_tokens": 0}

    def _wait_for_rate_limits(self):
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
            self._wait_for_rate_limits()

            response = openai.ChatCompletion.create(
                model=self.client.model_name,
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
