import torch as th

from jarl.data.core import MultiTensor
from jarl.data.types import SampleOutput
from jarl.train.sample.base import Sampler


class BatchSampler(Sampler):
    
    def __init__(
        self, 
        batch_size: int,
        num_epoch: int = 1,
        num_batch: int = None 
    ) -> None:
        super().__init__()
        self.batch_size = batch_size
        self.num_epoch = num_epoch
        self.num_batch = num_batch

    def sample(self, data: MultiTensor) -> SampleOutput:
        data = data.flatten(0, 1)  # flatten data into single-row tensors

        # calculate # of batches if not provided
        num_batch = self.num_batch or (len(data) // self.batch_size)

        for _ in range(self.num_epoch):
            # randomly shuffle indices
            idx = th.randperm(len(data))

            # yield slices of input data
            for i in range(0, num_batch * self.batch_size, self.batch_size):
                yield data[idx[i : i + self.batch_size]]