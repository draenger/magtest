from .dto.usage import Usage
from .utils.rate_limiter import RateLimiter


class BaseModel:
    def __init__(
        self,
        model_name,
        tokenizer,
        input_cost_per_million,
        output_cost_per_million,
        rpm_limit,
        rpd_limit,
        tpm_limit,
        tpd_limit,
    ):
        self.model_name = model_name
        self.tokenizer = tokenizer
        self.input_token_cost = input_cost_per_million / 1_000_000
        self.output_token_cost = output_cost_per_million / 1_000_000
        self.usage = Usage(0, 0)
        self.rate_limiter = RateLimiter(rpm_limit, rpd_limit, tpm_limit, tpd_limit)

    def get_model_name(self):
        return self.model_name

    def get_model_in_token_cost(self):
        return self.input_token_cost

    def get_model_out_token_cost(self):
        return self.output_token_cost

    def get_model_in_token_used(self):
        return self.usage.prompt_tokens

    def get_model_out_token_used(self):
        return self.usage.completion_tokens

    def estimate_tokens_amount(self, text):
        return len(self.tokenizer.encode(text))

    def update_usage(self, prompt_tokens: int, completion_tokens: int):
        self.usage.prompt_tokens += prompt_tokens
        self.usage.completion_tokens += completion_tokens
        self.rate_limiter.update_token_usage(prompt_tokens + completion_tokens)

    def wait_for_rate_limits(self):
        self.rate_limiter.wait_for_rate_limits(self.usage.total_tokens)

    def reset_usage(self):
        self.usage = Usage(0, 0)
