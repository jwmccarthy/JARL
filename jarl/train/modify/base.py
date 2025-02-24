from typing import Set
from abc import ABC, abstractmethod

from jarl.data.core import MultiTensor


class DataModifier(ABC):

    @property
    @abstractmethod
    def requires_keys(self) -> Set[str]:
        ...

    @property
    @abstractmethod
    def produces_keys(self) -> Set[str]:
        ...

    @abstractmethod
    def __call__(self, data: MultiTensor) -> MultiTensor:
        ...