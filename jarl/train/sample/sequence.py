import torch as th

from typing import Self

from jarl.data.multi import MultiArray, MultiIterable, MultiTensor
from jarl.data.types import Device, SampleOutput
from jarl.train.sample.base import Sampler


class SequenceSampler(Sampler):
    """Sample shuffled batches of contiguous time-major sequences."""

    def __init__(
        self,
        sequence_len: int,
        batch_len: int = None,
        num_batch: int = None,
        num_epoch: int = 1,
        device: Device = None,
    ) -> None:
        assert sequence_len > 0, "sequence_len must be positive"
        assert batch_len or num_batch, (
            "One of 'batch_len' or 'num_batch' must be provided."
        )
        super().__init__(device)
        self._sequence_len = sequence_len
        self._batch_len = batch_len
        self._num_batch = num_batch
        self._num_epoch = num_epoch
        self._data: MultiIterable = None

    def __call__(self, data: MultiIterable) -> Self:
        assert len(data.shape) >= 2, (
            "SequenceSampler expects data shaped [time, environment, ...]"
        )
        time, num_envs = data.shape[:2]
        assert time % self._sequence_len == 0, (
            f"time dimension {time} must be divisible by sequence_len "
            f"{self._sequence_len}"
        )

        num_chunks = time // self._sequence_len
        sequences = {}
        for key, value in data.items():
            tail = value.shape[2:]
            chunked = value.reshape(
                num_chunks, self._sequence_len, num_envs, *tail
            )
            sequences[key] = chunked.swapaxes(1, 2).reshape(
                num_chunks * num_envs, self._sequence_len, *tail
            )
        self._data = data.__class__(**sequences)
        return self

    def __iter__(self) -> Self:
        assert self._data is not None, "Data must be provided to Sampler."
        total = len(self._data)
        assert not self._batch_len or self._batch_len <= total, (
            "batch_len cannot exceed the number of sequences"
        )
        assert not self._num_batch or self._num_batch <= total, (
            "num_batch cannot exceed the number of sequences"
        )

        self._epoch = self._batch = 0
        self._index = th.randperm(total)
        self._new_nb = self._num_batch or total // self._batch_len
        self._new_bl = self._batch_len or total // self._num_batch
        return self

    def __next__(self) -> SampleOutput:
        if self._batch >= self._new_nb:
            self._epoch += 1
            self._batch = 0
            self._index = th.randperm(len(self._data))

        if self._epoch >= self._num_epoch:
            raise StopIteration

        left = self._batch * self._new_bl
        right = left + self._new_bl
        index = self._index[left:right]
        if isinstance(self._data, MultiArray):
            index = index.numpy()

        # Recurrent modules consume time-major [sequence, batch, ...] data.
        batch = self._data.__class__(
            **{
                key: value[index].swapaxes(0, 1)
                for key, value in self._data.items()
            }
        )
        self._batch += 1

        return (
            MultiTensor.from_numpy(batch, device=self._device)
            if isinstance(batch, MultiArray) and self._device
            else batch
        )
