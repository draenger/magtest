from abc import ABC, abstractmethod


class SingleRunnerInterface(ABC):
    @abstractmethod
    def __init__(self, client, rpm_limit, rpd_limit, tpm_limit, tpd_limit):
        pass

    @abstractmethod
    def predict(self, prompt):
        pass

    @abstractmethod
    def get_in_token_used(self):
        pass

    @abstractmethod
    def get_out_token_used(self):
        pass
