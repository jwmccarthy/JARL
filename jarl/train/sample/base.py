from typing import Self
from abc import ABC, abstractmethod

from jarl.data.multi import MultiIterable
from jarl.data.types import Device, SampleOutput


class Sampler(ABC):

    def __init__(self, device: Device = None) -> None:
        self._device = device

    @abstractmethod
    def __call__(self, data: MultiIterable) -> Self:
        ...

    @abstractmethod
    def __iter__(self) -> Self:
        ...

    @abstractmethod
    def __next__(self) -> SampleOutput:
        ...