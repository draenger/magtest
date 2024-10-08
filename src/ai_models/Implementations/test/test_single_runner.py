import random
import time
from ...interfaces import SingleRunnerInterface


class TestSingleRunner(SingleRunnerInterface):
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
        self.in_token_used = 0
        self.out_token_used = 0

    def _wait_for_rate_limits(self):
        current_time = time.time()

        # Reset daily counters if a new day has started
        if current_time - self.last_reset_time >= 86400:  # 24 hours in seconds
            self.daily_token_usage = 0
            self.daily_requests = 0
            self.last_reset_time = current_time

        # Manage RPM limit
        self.request_times = [t for t in self.request_times if current_time - t < 60]
        if len(self.request_times) >= self.rpm_limit:
            time.sleep(0.1)  # Small delay instead of exception

        # Manage RPD limit
        if self.rpd_limit and self.daily_requests >= self.rpd_limit:
            time.sleep(0.1)  # Small delay instead of exception

        # Manage TPM limit
        if self.tpm_limit and self.token_usage >= self.tpm_limit:
            time.sleep(0.1)  # Small delay instead of exception

        # Manage TPD limit
        if self.tpd_limit and self.daily_token_usage >= self.tpd_limit:
            time.sleep(0.1)  # Small delay instead of exception

        self.request_times.append(current_time)
        self.daily_requests += 1

    def predict(self, prompt):
        self._wait_for_rate_limits()
        in_tokens = self.client.estimate_tokens_ammount(prompt)
        out_tokens = 1  # Assuming 1 token for output in this test implementation
        self._update_token_usage(in_tokens, out_tokens)
        return random.choice(["A", "B", "C", "D"])

    def _update_token_usage(self, in_tokens, out_tokens):
        self.token_usage += in_tokens + out_tokens
        self.daily_token_usage += in_tokens + out_tokens
        self.in_token_used += in_tokens
        self.out_token_used += out_tokens

    def get_in_token_used(self):
        return self.in_token_used

    def get_out_token_used(self):
        return self.out_token_used
