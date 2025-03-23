import torch as th

from typing import Self

from jarl.data.types import Device
from jarl.data.types import SampleOutput
from jarl.data.multi import MultiIterable, MultiTensor

from jarl.train.sample.base import Sampler


class BatchSampler(Sampler):

    def __init__(
        self,
        batch_len: int = None,
        num_batch: int = None,
        num_epoch: int = 1,
        device: Device = None
    ) -> None:
        # 2 of 3 must be provided
        assert batch_len or num_batch, (
            "One of 'batch_len' or 'num_batch' must be provided."
        )

        super().__init__(device)
        self._batch_len = batch_len
        self._num_epoch = num_epoch
        self._num_batch = num_batch
        self._data: MultiIterable = None

    def __call__(self, data: MultiIterable) -> Self:
        self._data = data.flatten(0, 1)
        return self
    
    def __iter__(self) -> Self:
        assert self._data, "Data must be provided to Sampler."

        self._epoch = self._batch = 0
        self._index = th.randperm(len(self._data))

        # compute num_batch
        self._new_nb = (self._num_batch
            or len(self._data) // self._batch_len)

        # compute batch_len
        self._new_bl = self._batch_len \
            or len(self._data) // self._num_batch

        return self
    
    def __next__(self) -> SampleOutput:  
        if self._batch >= self._new_nb:
            self._epoch += 1
            self._batch = 0
            self._index = th.randperm(len(self._data))

        if self._epoch >= self._num_epoch:
            raise StopIteration
        
        # generate batch from random index
        l_idx = self._batch * self._new_bl
        r_idx = l_idx + self._new_bl
        batch = self._data[self._index[l_idx:r_idx]]

        self._batch += 1

        return (
            MultiTensor.from_numpy(batch, device=self._device)
            if self._device else batch
        )