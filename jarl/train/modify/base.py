from typing import Set
from abc import ABC, abstractmethod

from jarl.data.core import MultiTensor


class DataModifier(ABC):

    _requires_keys: Set[str]
    _produces_keys: Set[str]

    @property
    def requires_keys(self) -> Set[str]:
        return self._requires_keys
    
    @property
    def produces_keys(self) -> Set[str]:
        return self._produces_keys

    @abstractmethod
    def __call__(self, data: MultiTensor) -> MultiTensor:
        ...