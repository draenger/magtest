import random
from .ai_model import AIModel


class TestModel(AIModel):
    def __init__(self, model_name="TestModel"):
        self.model_name = model_name
        self.tokens_used = random.randint(0, 100)  # Randomly assign tokens used

    def predict(self, prompt):
        return random.choice(["A", "B", "C", "D"])  # Randomly select an answer

    def get_tokens_used(self):
        return self.tokens_used
