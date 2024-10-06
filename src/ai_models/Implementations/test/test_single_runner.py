import random
import time


class TestSingleRunner:
    def __init__(self, client, rpm_limit, tpm_limit, tpd_limit):
        self.client = client
        self.rpm_limit = rpm_limit
        self.tpm_limit = tpm_limit
        self.tpd_limit = tpd_limit
        self.request_times = []
        self._token_usage = 0
        self._daily_token_usage = 0
        self._in_token_used = 0
        self._out_token_used = 0
        self.last_reset_time = time.time()

    def _wait_for_rate_limits(self):
        current_time = time.time()
        self.request_times = [t for t in self.request_times if current_time - t < 60]
        if len(self.request_times) >= self.rpm_limit:
            time.sleep(0.1)  # Small delay instead of exception
        if self.tpm_limit and self._token_usage >= self.tpm_limit:
            time.sleep(0.1)  # Small delay instead of exception
        if self.tpd_limit and self._daily_token_usage >= self.tpd_limit:
            time.sleep(0.1)  # Small delay instead of exception
        self.request_times.append(current_time)

    def predict(self, prompt):
        self._wait_for_rate_limits()
        in_tokens = self.client.estimate_tokens_ammount(prompt)
        out_tokens = 1  # Assuming 1 token for output in this test implementation
        self._update_token_usage(in_tokens, out_tokens)
        return random.choice(["A", "B", "C", "D"])

    def _update_token_usage(self, in_tokens, out_tokens):
        self._token_usage += in_tokens + out_tokens
        self._daily_token_usage += in_tokens + out_tokens
        self._in_token_used += in_tokens
        self._out_token_used += out_tokens

    def get_in_token_used(self):
        return self._in_token_used

    def get_out_token_used(self):
        return self._out_token_used

    def get_token_usage(self):
        return self._token_usage

    def get_daily_token_usage(self):
        return self._daily_token_usage

    def reset_token_usage(self):
        self._token_usage = 0
        self._daily_token_usage = 0
        self._in_token_used = 0
        self._out_token_used = 0
        self.last_reset_time = time.time()

    def estimate_tokens_amount(self, text):
        return self.client.estimate_tokens_ammount(text)
