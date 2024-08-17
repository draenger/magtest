from abc import ABC, abstractmethod

class AIModel(ABC):
    @abstractmethod
    def predict(self, prompt):
        pass

    @abstractmethod
    def get_tokens_used(self):
        pass
