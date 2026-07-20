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
        learner_mask = flat.get("learner_mask")
        if learner_mask is not None:
            flat = flat[learner_mask.bool()]
            if not len(flat):
                raise RuntimeError("rollout contains no learner transitions")

        for _ in range(self.epochs):
            yield from self._sample_epoch(flat)

    def _sample_epoch(self, data: TensorBatch):
        indices = th.randperm(len(data), device=data.device)

        batch_size = min(self.batch_size, len(data))
        for left in range(0, len(data), batch_size):
            selected = indices[left : left + batch_size]

            if len(selected):
                yield data[selected]


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
        fields: tuple[str, ...] | None = None,
    ) -> None:
        if sequence_length < 1 or sequences_per_batch < 1 or epochs < 1:
            raise ValueError("sequence settings must be positive")

        self.sequence_length = sequence_length
        self.sequences_per_batch = sequences_per_batch
        self.epochs = epochs
        self.fields = fields

    def __call__(self, data: TensorBatch):
        if "policy_state" not in data:
            raise ValueError("recurrent sampling requires policy_state")
        if self.fields is not None:
            required = list(self.fields)
            for key in (
                "policy_state",
                "terminated",
                "truncated",
                "learner_mask",
            ):
                if key in data and key not in required:
                    required.append(key)
            data = data.select(*required)

        time, num_envs = data.shape[:2]

        chunks = time // self.sequence_length
        if not chunks:
            raise ValueError("rollout is shorter than one recurrent sequence")
        if time % self.sequence_length:
            data = data[: chunks * self.sequence_length]
        sequences = self._build_sequences(data, chunks, num_envs)
        learner_mask = sequences.get("learner_mask")
        if learner_mask is None:
            eligible = th.arange(chunks * num_envs, device=data.device)
        else:
            eligible = learner_mask.any(dim=1).nonzero(as_tuple=True)[0]
        if not len(eligible):
            raise RuntimeError("rollout contains no learner sequences")
        done = sequences["terminated"] | sequences["truncated"]
        has_reset = done[:, :-1].any(dim=1)
        clean = eligible[~has_reset[eligible]]
        resetting = eligible[has_reset[eligible]]

        for _ in range(self.epochs):
            yield from self._sample_epoch(sequences, clean)
            yield from self._sample_epoch(sequences, resetting)

    def _build_sequences(
        self,
        data: TensorBatch,
        chunks: int,
        num_envs: int,
    ) -> dict[str, th.Tensor]:
        sequence_count = chunks * num_envs
        sequences = {}

        for key, value in data.items():
            tail = value.shape[2:]
            sequences[key] = (
                value.reshape(chunks, self.sequence_length, num_envs, *tail)
                .swapaxes(1, 2)
                .reshape(sequence_count, self.sequence_length, *tail)
            )

        return sequences

    def _sample_epoch(
        self,
        sequences: dict[str, th.Tensor],
        eligible: th.Tensor,
    ):
        if not len(eligible):
            return
        device = next(iter(sequences.values())).device
        indices = eligible[th.randperm(len(eligible), device=device)]

        for left in range(0, len(indices), self.sequences_per_batch):
            selected = indices[left : left + self.sequences_per_batch]
            if len(selected):
                yield self._build_batch(sequences, selected)

    @staticmethod
    def _build_batch(
        sequences: dict[str, th.Tensor],
        selected: th.Tensor,
    ) -> SequenceBatch:
        state = sequences["policy_state"][selected, 0]
        step_data = {
            key: value[selected].swapaxes(0, 1)
            for key, value in sequences.items()
            if key != "policy_state"
        }
        steps = TensorBatch(step_data)

        done = steps["terminated"] | steps["truncated"]
        reset = th.zeros_like(done)
        reset[1:] = done[:-1]
        valid = steps.get("learner_mask")
        if valid is None:
            valid = th.ones_like(done)

        return SequenceBatch(
            steps=steps,
            initial_state=state,
            reset=reset,
            valid=valid.bool(),
        )
