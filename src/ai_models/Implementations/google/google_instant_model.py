# google_instant_model.py
from ...base_model import BaseModel
from ...interfaces.instant_model_interface import InstantModelInterface
from ...dto.usage import Usage
from ...dto.instant_response import InstantResponse
from vertexai.language_models import TextGenerationModel
import vertexai
import os


class GoogleInstantModel(BaseModel, InstantModelInterface):
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
        vertexai.init(
            project=os.getenv("GOOGLE_CLOUD_PROJECT"),
            location=os.getenv("GOOGLE_CLOUD_LOCATION"),
        )
        # self.model = TextGenerationModel.from_pretrained(model_name)

    def predict(self, prompt: str, max_tokens: int = 1) -> InstantResponse:
        try:
            self.wait_for_rate_limits()

            response = self.model.predict(
                prompt=prompt, max_output_tokens=max_tokens, temperature=0.5
            )

            # Oszacuj użycie tokenów
            input_tokens = self.estimate_tokens_amount(prompt)
            output_tokens = self.estimate_tokens_amount(response.text)

            usage = Usage(prompt_tokens=input_tokens, completion_tokens=output_tokens)

            self.update_usage(input_tokens, output_tokens)

            return InstantResponse(prediction=response.text, usage=usage)

        except Exception as e:
            print(f"Error in prediction: {e}")
            return InstantResponse(prediction=None, usage=Usage())

    def update_usage(self, prompt_tokens: int, completion_tokens: int):
        super().update_usage(prompt_tokens, completion_tokens)
