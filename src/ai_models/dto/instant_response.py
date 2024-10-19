from dataclasses import dataclass
from .usage import Usage


@dataclass
class InstantResponse:
    prediction: str
    usage: Usage
