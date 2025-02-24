from abc import ABC, abstractmethod

from jarl.data.core import MultiTensor
from jarl.data.types import SampleOutput


class Sampler(ABC):

    @abstractmethod
    def sample(self, data: MultiTensor) -> SampleOutput:
        ...


class IdentitySampler(Sampler):

    def sample(self, data: MultiTensor) -> SampleOutput:
        return (data,)