from dataclasses import dataclass

import torch as th

from jarl.data.batch import TensorBatch


class RolloutMinibatches:
    def __init__(self, batch_size: int, epochs: int = 1) -> None:
        if batch_size < 1 or epochs < 1:
            raise ValueError("minibatch settings must be positive")
        self.batch_size = batch_size
        self.epochs = epochs

    def __call__(self, data: TensorBatch):
        if len(data.shape) < 2:
            raise ValueError("rollout data must be [time, environment, ...]")
        flat = data.flatten(0, 1)
        if self.batch_size > len(flat):
            raise ValueError("batch size exceeds rollout size")
        for _ in range(self.epochs):
            indices = th.randperm(len(flat), device=flat.device)
            for left in range(0, len(flat), self.batch_size):
                selected = indices[left : left + self.batch_size]
                if len(selected):
                    yield flat[selected]


@dataclass(frozen=True)
class SequenceBatch:
    steps: TensorBatch
    initial_state: th.Tensor
    reset: th.Tensor
    valid: th.Tensor


class RecurrentRolloutMinibatches:
    def __init__(
        self,
        sequence_length: int,
        sequences_per_batch: int,
        epochs: int = 1,
    ) -> None:
        if sequence_length < 1 or sequences_per_batch < 1 or epochs < 1:
            raise ValueError("sequence settings must be positive")
        self.sequence_length = sequence_length
        self.sequences_per_batch = sequences_per_batch
        self.epochs = epochs

    def __call__(self, data: TensorBatch):
        if "behavior_state" not in data:
            raise ValueError("recurrent sampling requires behavior_state")
        time, num_envs = data.shape[:2]
        if time % self.sequence_length:
            raise ValueError("rollout time must be divisible by sequence length")
        chunks = time // self.sequence_length
        sequence_count = chunks * num_envs
        if self.sequences_per_batch > sequence_count:
            raise ValueError("batch requests more sequences than the rollout")

        sequences = {}
        for key, value in data.items():
            tail = value.shape[2:]
            sequences[key] = (
                value.reshape(chunks, self.sequence_length, num_envs, *tail)
                .swapaxes(1, 2)
                .reshape(sequence_count, self.sequence_length, *tail)
            )

        for _ in range(self.epochs):
            indices = th.randperm(sequence_count, device=data.device)
            for left in range(0, sequence_count, self.sequences_per_batch):
                selected = indices[left : left + self.sequences_per_batch]
                if len(selected) != self.sequences_per_batch:
                    continue
                state = sequences["behavior_state"][selected, 0]
                step_data = {
                    key: value[selected].swapaxes(0, 1)
                    for key, value in sequences.items()
                    if key != "behavior_state"
                }
                steps = TensorBatch(step_data)
                done = steps["terminated"] | steps["truncated"]
                reset = th.zeros_like(done)
                reset[1:] = done[:-1]
                yield SequenceBatch(
                    steps=steps,
                    initial_state=state,
                    reset=reset,
                    valid=th.ones_like(done),
                )
