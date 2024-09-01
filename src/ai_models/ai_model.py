from abc import ABC, abstractmethod


class AIModel(ABC):

    @abstractmethod
    def get_model_name(self):
        pass

    @abstractmethod
    def get_model_in_token_cost(self):
        pass

    @abstractmethod
    def get_model_out_token_cost(self):
        pass

    @abstractmethod
    def get_model_in_token_used(self):
        pass

    @abstractmethod
    def get_model_out_token_used(self):
        pass

    @abstractmethod
    def predict(self, prompt):
        pass

    @abstractmethod
    def estimate_tokens_ammount(self, text):
        pass
