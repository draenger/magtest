import time
import openai
from threading import Lock
from ...interfaces import SingleRunnerInterface


class OpenAISingleRunner(SingleRunnerInterface):
    def __init__(self, client, rpm_limit, rpd_limit, tpm_limit, tpd_limit):
        self.client = client
        self.rpm_limit = rpm_limit
        self.rpd_limit = rpd_limit
        self.tpm_limit = tpm_limit
        self.tpd_limit = tpd_limit
        self.request_times = []
        self.daily_requests = 0
        self.token_usage = 0
        self.daily_token_usage = 0
        self.last_reset_time = time.time()
        self.lock = Lock()
        self.usage = {"prompt_tokens": 0, "completion_tokens": 0}

    def _wait_for_rate_limits(self):
        with self.lock:
            current_time = time.time()

            # Reset daily counters if a new day has started
            if current_time - self.last_reset_time >= 86400:  # 24 hours in seconds
                self.daily_token_usage = 0
                self.daily_requests = 0
                self.last_reset_time = current_time

            # Manage RPM limit
            self.request_times = [
                t for t in self.request_times if current_time - t < 60
            ]
            if len(self.request_times) >= self.rpm_limit:
                sleep_time = 60 - (current_time - self.request_times[0])
                if sleep_time > 0:
                    time.sleep(sleep_time)

            # Manage RPD limit
            if self.rpd_limit and self.daily_requests >= self.rpd_limit:
                sleep_time = self.last_reset_time + 86400 - current_time
                if sleep_time > 0:
                    time.sleep(sleep_time)

            # Manage TPM limit
            if self.tpm_limit and self.token_usage >= self.tpm_limit:
                sleep_time = 60 - (current_time - self.request_times[0])
                if sleep_time > 0:
                    time.sleep(sleep_time)
                self.token_usage = 0

            # Manage TPD limit
            if self.tpd_limit and self.daily_token_usage >= self.tpd_limit:
                sleep_time = self.last_reset_time + 86400 - current_time
                if sleep_time > 0:
                    time.sleep(sleep_time)

            self.request_times.append(current_time)
            self.daily_requests += 1

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
