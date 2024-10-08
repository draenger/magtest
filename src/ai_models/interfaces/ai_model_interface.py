from abc import ABC, abstractmethod


class AIModelInterface(ABC):

    @abstractmethod
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
        batch_queue_limit,
    ):
        pass

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

    @abstractmethod
    def add_batch_request(self, custom_id, model, messages, max_tokens=1):
        pass

    @abstractmethod
    def run_batch(self, benchmark_name, metadata=None):
        pass

    @abstractmethod
    def check_batch_results(self, benchmark_name, batch_id):
        pass
