import time
from threading import Lock


class RateLimiter:
    def __init__(self, rpm_limit, rpd_limit, tpm_limit, tpd_limit):
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

    def wait_for_rate_limits(self, tokens):
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
            if self.tpm_limit and self.token_usage + tokens > self.tpm_limit:
                sleep_time = 60 - (current_time - self.request_times[0])
                if sleep_time > 0:
                    time.sleep(sleep_time)
                self.token_usage = 0

            # Manage TPD limit
            if self.tpd_limit and self.daily_token_usage + tokens > self.tpd_limit:
                sleep_time = self.last_reset_time + 86400 - current_time
                if sleep_time > 0:
                    time.sleep(sleep_time)

            self.request_times.append(current_time)
            self.daily_requests += 1

    def update_token_usage(self, tokens):
        self.token_usage += tokens
        self.daily_token_usage += tokens
