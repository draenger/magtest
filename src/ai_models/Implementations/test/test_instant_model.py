from ...base_model import BaseModel
from ...interfaces.instant_model_interface import InstantModelInterface
from ...dto.usage import Usage
from ...dto.instant_response import InstantResponse
import random


class TestInstantModel(BaseModel, InstantModelInterface):
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

    def predict(self, prompt: str, max_tokens: int = 1) -> InstantResponse:
        self.wait_for_rate_limits()
        prediction = random.choice(["A", "B", "C", "D"])
        usage = Usage(
            prompt_tokens=self.estimate_tokens_amount(prompt), completion_tokens=1
        )
        self.update_usage(usage.prompt_tokens, usage.completion_tokens)
        return InstantResponse(prediction=prediction, usage=usage)

    # Metody z BaseModel są już dostępne dzięki dziedziczeniu,
    # ale możemy je nadpisać, jeśli potrzebujemy specyficznej implementacji dla TestInstantModel

    def estimate_tokens_amount(self, text: str) -> int:
        # Możemy użyć implementacji z BaseModel lub zdefiniować własną
        return super().estimate_tokens_amount(text)

    # Inne metody specyficzne dla TestInstantModel można dodać tutaj
