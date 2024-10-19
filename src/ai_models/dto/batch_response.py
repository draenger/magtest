from dataclasses import dataclass
from typing import Optional
from .usage import Usage


@dataclass
class BatchResponseItem:
    custom_id: str
    response: Optional[str]
    usage: Optional[Usage]
    status: str


class BatchResponse:
    def __init__(self, items: list[BatchResponseItem]):
        self.items = items

    def __iter__(self):
        return iter(self.items)

    def __getitem__(self, index):
        return self.items[index]

    def __len__(self):
        return len(self.items)
