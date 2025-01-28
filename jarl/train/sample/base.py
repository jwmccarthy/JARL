from typing import Iterator
from abc import ABC, abstractmethod

from jarl.data.multi import MultiTensor


class BatchSampler:
    
    def __init__(
        self, 
        batch_size: int,
        num_batches: int = None 
    ) -> None:
        self.batch_size = batch_size
        self.num_batches = num_batches

    @abstractmethod
    def sample(self) -> Iterator[MultiTensor]:
        ...