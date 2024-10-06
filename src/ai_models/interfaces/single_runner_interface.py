from abc import ABC, abstractmethod


class SingleRunnerInterface(ABC):
    @abstractmethod
    def __init__(self, client, rpm_limit, tpm_limit=None, tpd_limit=None):
        pass

    @abstractmethod
    def predict(self, prompt):
        pass
