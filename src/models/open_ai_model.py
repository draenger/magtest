import os
import openai
from dotenv import load_dotenv
from .ai_model import AIModel


class OpenAIModel(AIModel):
    def __init__(self, model_name):
        self.model_name = model_name
        self.tokens_used = 0
        load_dotenv()
        openai.api_key = os.getenv("OPENAI_API_KEY")

    def predict(self, prompt):
        try:
            self.tokens_used = 0
            response = openai.ChatCompletion.create(
                model=self.model_name,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a helpful assistant that answers multiple choice questions.",
                    },
                    {"role": "user", "content": prompt},
                ],
                max_tokens=1,
                n=1,
                stop=None,
                temperature=0.5,
            )

            self.tokens_used += response["usage"]["total_tokens"]
            answer = response.choices[0].message["content"].strip().upper()
            if answer in ["A", "B", "C", "D"]:
                return answer
            else:
                return None
        except Exception as e:
            print(f"Error in prediction: {e}")
            return None

    def get_tokens_used(self):
        return self.tokens_used
