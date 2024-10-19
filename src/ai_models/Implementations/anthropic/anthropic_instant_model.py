from ...base_model import BaseModel
from ...interfaces.instant_model_interface import InstantModelInterface
from ...dto.usage import Usage
from ...dto.instant_response import InstantResponse
from anthropic import Anthropic


class AnthropicInstantModel(BaseModel, InstantModelInterface):
    def __init__(
        self,
        model_name: str,
        tokenizer,
        input_cost_per_million: float,
        output_cost_per_million: float,
        rpm_limit: int,
        rpd_limit: int,
        tpm_limit: int,
        tpd_limit: int,
    ):
        super().__init__(
            model_name,
            tokenizer,
            input_cost_per_million,
            output_cost_per_million,
            rpm_limit,
            rpd_limit,
            tpm_limit,
            tpd_limit,
        )
        self.anthropic_client = Anthropic()

    def get_model_name(self) -> str:
        return super().get_model_name()

    def get_model_in_token_cost(self) -> float:
        return super().get_model_in_token_cost()

    def get_model_out_token_cost(self) -> float:
        return super().get_model_out_token_cost()

    def get_model_in_token_used(self) -> int:
        return super().get_model_in_token_used()

    def get_model_out_token_used(self) -> int:
        return super().get_model_out_token_used()

    def estimate_tokens_amount(self, text: str) -> int:
        return super().estimate_tokens_amount(text)

    def wait_for_rate_limits(self):
        super().wait_for_rate_limits()

    def reset_usage(self):
        super().reset_usage()

    def predict(self, prompt: str, max_tokens: int = 1) -> InstantResponse:
        try:
            prediction_usage = Usage()
            self.wait_for_rate_limits()

            response = self.anthropic_client.completions.create(
                model=self.model_name,
                prompt=prompt,
                max_tokens_to_sample=max_tokens,
            )

            prediction_usage.prompt_tokens = response.usage.prompt_tokens
            prediction_usage.completion_tokens = response.usage.completion_tokens

            self.update_usage(
                prediction_usage.prompt_tokens, prediction_usage.completion_tokens
            )

            return InstantResponse(
                prediction=response.completion, usage=prediction_usage
            )
        except Exception as e:
            print(f"Error in prediction: {e}")
            return InstantResponse(prediction=None, usage=Usage())

    def update_usage(self, prompt_tokens: int, completion_tokens: int):
        super().update_usage(prompt_tokens, completion_tokens)
