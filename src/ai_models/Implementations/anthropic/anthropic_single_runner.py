import time
from anthropic import Anthropic
from threading import Lock
from ...interfaces import SingleRunnerInterface


class AnthropicSingleRunner(SingleRunnerInterface):
    def __init__(self, client, rpm_limit, tpm_limit, tpd_limit):
        self.client = client
        self.rpm_limit = rpm_limit
        self.tpm_limit = tpm_limit
        self.tpd_limit = tpd_limit
        self.request_times = []
        self.token_usage = 0
        self.daily_token_usage = 0
        self.last_reset_time = time.time()
        self.lock = Lock()
        self.anthropic_client = Anthropic(api_key=client.api_key)
        self.in_token_used = 0
        self.out_token_used = 0

    def _wait_for_rate_limits(self):
        with self.lock:
            current_time = time.time()

            # Reset daily token usage if a new day has started
            if current_time - self.last_reset_time >= 86400:  # 24 hours in seconds
                self.daily_token_usage = 0
                self.last_reset_time = current_time

            # Manage RPM limit
            self.request_times = [
                t for t in self.request_times if current_time - t < 60
            ]
            if len(self.request_times) >= self.rpm_limit:
                sleep_time = 60 - (current_time - self.request_times[0])
                if sleep_time > 0:
                    time.sleep(sleep_time)

            # Manage TPM limit
            if self.token_usage >= self.tpm_limit:
                sleep_time = 60 - (current_time - self.request_times[0])
                if sleep_time > 0:
                    time.sleep(sleep_time)
                self.token_usage = 0

            # Manage TPD limit
            if self.daily_token_usage >= self.tpd_limit:
                sleep_time = self.last_reset_time + 86400 - current_time
                if sleep_time > 0:
                    time.sleep(sleep_time)
                self.daily_token_usage = 0

            self.request_times.append(current_time)

    def predict(self, prompt):
        try:
            self._wait_for_rate_limits()
            response = self.anthropic_client.completions.create(
                model=self.client.model_name, prompt=prompt, max_tokens_to_sample=1
            )
            self.in_token_used = response.usage.prompt_tokens
            self.out_token_used = response.usage.completion_tokens
            tokens_used = response.usage.total_tokens
            self.token_usage += tokens_used
            self.daily_token_usage += tokens_used
            return response.completion
        except Exception as e:
            print(f"Error in prediction: {e}")
            return None

    def get_in_token_used(self):
        return self.in_token_used

    def get_out_token_used(self):
        return self.out_token_used
