from dataclasses import dataclass

import torch as th

from jarl.data.batch import TensorBatch


class RolloutMinibatches:
    def __init__(self, batch_size: int, epochs: int = 1) -> None:
        if batch_size < 1 or epochs < 1:
            raise ValueError("minibatch settings must be positive")

        self.batch_size = batch_size
        self.epochs = epochs
        self._epoch_callback = None

    def set_epoch_callback(self, callback) -> None:
        self._epoch_callback = callback

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
            if self._epoch_callback is not None:
                self._epoch_callback()

    def _sample_epoch(self, data: TensorBatch):
        indices = th.randperm(len(data), device=data.device)

        batch_size = min(self.batch_size, len(data))
        for left in range(0, len(data), batch_size):
            selected = indices[left : left + batch_size]

            if len(selected):
                yield data[selected]


@dataclass(frozen=True)
class SequenceBatch:
    steps:                TensorBatch
    initial_state:        th.Tensor
    reset:                th.Tensor | None
    valid:                th.Tensor
    initial_critic_state: th.Tensor | None = None


@dataclass(frozen=True)
class ChunkBatch:
    data:             TensorBatch
    valid:            th.Tensor
    duration:         th.Tensor
    planned_duration: th.Tensor


class TrajectoryChunkMinibatches:
    def __init__(
        self,
        horizon:     int,
        jitter:      int,
        batch_size:  int,
        epochs:      int = 1,
    ) -> None:
        if horizon < 1 or jitter < 0 or batch_size < 1 or epochs < 1:
            raise ValueError("chunk settings must be positive")

        self.horizon = horizon
        self.jitter = jitter
        self.batch_size = batch_size
        self.epochs = epochs
        self.max_duration = horizon + jitter
        if batch_size < self.max_duration:
            raise ValueError("batch size must fit the longest chunk")
        self._epoch_callback = None

    def set_epoch_callback(self, callback) -> None:
        self._epoch_callback = callback

    def __call__(self, data: TensorBatch):
        if len(data.shape) < 2:
            raise ValueError("rollout data must be [time, environment, ...]")

        time, num_envs = data.shape[:2]
        done = (data["terminated"] | data["truncated"]).swapaxes(0, 1).cpu()

        for _ in range(self.epochs):
            chunks = []
            for env in range(num_envs):
                chunks.extend(self._chunk_env(done[env], env, time))

            if not chunks:
                raise RuntimeError("rollout contains no chunks")

            order = th.randperm(len(chunks)).tolist()
            yield from self._pack_batches(data, chunks, order)

            if self._epoch_callback is not None:
                self._epoch_callback()

    def _chunk_env(
        self,
        done: th.Tensor,
        env:  int,
        time: int,
    ) -> list[tuple[int, int, int, int, int]]:
        boundaries = done.nonzero(as_tuple=True)[0].tolist()

        if not boundaries or boundaries[-1] != time - 1:
            boundaries.append(time - 1)

        boundaries = sorted(set(boundaries))

        chunks = []
        start = 0
        for boundary in boundaries:
            while start <= boundary:
                remaining = boundary - start + 1
                low = max(1, self.horizon - self.jitter)
                high = self.horizon + self.jitter
                planned = int(th.randint(low, high + 1, (1,)).item())
                duration = min(planned, remaining)
                end = min(start + duration, boundary + 1)
                duration = end - start

                chunks.append((env, start, end, duration, planned))
                start = end

            start = boundary + 1

        return chunks

    def _pack_batches(
        self,
        data:   TensorBatch,
        chunks: list[tuple[int, int, int, int, int]],
        order:  list[int],
    ):
        selected = []
        valid_steps = 0

        for index in order:
            env, start, end, duration, planned = chunks[index]

            if valid_steps + duration > self.batch_size and selected:
                yield self._build_batch(data, selected)
                selected = []
                valid_steps = 0

            selected.append((env, start, end, duration, planned))
            valid_steps += duration

        if selected:
            yield self._build_batch(data, selected)

    def _build_batch(
        self,
        data:    TensorBatch,
        chunks:  list[tuple[int, int, int, int, int]],
    ) -> ChunkBatch:
        max_duration = self.max_duration
        device = data.device
        environments, starts, _, durations, planned = zip(*chunks)
        environments = th.tensor(environments, dtype=th.long, device=device)[:, None]
        starts = th.tensor(starts, dtype=th.long, device=device)[:, None]
        duration = th.tensor(durations, dtype=th.long, device=device)
        planned_duration = th.tensor(planned, dtype=th.long, device=device)
        offsets = th.arange(max_duration, device=device)[None, :]
        valid = offsets < duration[:, None]
        time_index = (starts + offsets).clamp_max(data.shape[0] - 1)

        batch = {}
        for key, value in data.items():
            gathered = value[time_index, environments]
            mask = valid.view(*valid.shape, *((1,) * (gathered.ndim - 2)))
            batch[key] = th.where(mask, gathered, th.zeros_like(gathered))

        return ChunkBatch(
            data=TensorBatch(batch),
            valid=valid,
            duration=duration,
            planned_duration=planned_duration,
        )


class RecurrentRolloutMinibatches:
    required_fields = (
        "policy_state",
        "critic_state",
        "terminated",
        "truncated",
        "learner_mask",
        "valid",
    )

    def __init__(
        self,
        sequence_length:     int,
        sequences_per_batch: int,
        epochs:              int = 1,
        fields:              tuple[str, ...] | None = None,
    ) -> None:
        if sequence_length < 1 or sequences_per_batch < 1 or epochs < 1:
            raise ValueError("sequence settings must be positive")

        self.sequence_length = sequence_length
        self.sequences_per_batch = sequences_per_batch
        self.epochs = epochs
        self.fields = fields
        self._epoch_callback = None

    def set_epoch_callback(self, callback) -> None:
        self._epoch_callback = callback

    def __call__(self, data: TensorBatch):
        if "policy_state" not in data:
            raise ValueError("recurrent sampling requires policy_state")

        data = self._select_fields(data)
        data = self._pad_rollout(data)
        time, num_envs = data.shape[:2]

        chunks = time // self.sequence_length
        sequences = self._build_sequences(data, chunks, num_envs)
        structural_valid = self._structural_valid(sequences, chunks, num_envs)
        combined_valid = self._combine_valid(sequences, chunks, num_envs)

        eligible = combined_valid.any(dim=1).nonzero(as_tuple=True)[0]
        if not len(eligible):
            raise RuntimeError("rollout contains no valid sequences")

        batch_sizes = self._batch_sizes(len(eligible))

        done = (
            sequences["terminated"] | sequences["truncated"]
        ) & structural_valid
        has_reset = done[:, :-1].any(dim=1)
        clean = eligible[~has_reset[eligible]]
        resetting = eligible[has_reset[eligible]]

        for _ in range(self.epochs):
            yield from self._sample_epoch(
                sequences,
                clean,
                resetting,
                batch_sizes,
                combined_valid,
                structural_valid,
            )
            if self._epoch_callback is not None:
                self._epoch_callback()

    def _select_fields(self, data: TensorBatch) -> TensorBatch:
        if self.fields is None:
            return data

        required = list(self.fields)
        required.extend(
            key
            for key in self.required_fields
            if key in data and key not in required
        )
        return data.select(*required)

    def _pad_rollout(self, data: TensorBatch) -> TensorBatch:
        time, num_envs = data.shape[:2]
        padding = -time % self.sequence_length
        if not padding:
            return data

        had_learner_mask = "learner_mask" in data
        data = TensorBatch(
            {
                key: self._pad_tensor(value, padding)
                for key, value in data.items()
            }
        )

        if had_learner_mask:
            return data

        valid = th.ones(
            (time + padding, num_envs),
            dtype=th.bool,
            device=data.device,
        )
        valid[time:] = False

        return data.with_fields(learner_mask=valid)

    @staticmethod
    def _pad_tensor(value: th.Tensor, padding: int) -> th.Tensor:
        zeros = th.zeros(
            (padding, *value.shape[1:]),
            dtype=value.dtype,
            device=value.device,
        )
        return th.cat((value, zeros))

    def _build_sequences(
        self,
        data:     TensorBatch,
        chunks:   int,
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

    def _combine_valid(
        self,
        sequences: dict[str, th.Tensor],
        chunks:    int,
        num_envs:  int,
    ) -> th.Tensor:
        device = next(iter(sequences.values())).device
        learner_mask = sequences.get("learner_mask")
        valid = sequences.get("valid")

        if learner_mask is None and valid is None:
            return th.ones(
                chunks * num_envs,
                self.sequence_length,
                dtype=th.bool,
                device=device,
            )
        if learner_mask is None:
            return valid.bool()
        if valid is None:
            return learner_mask.bool()
        return learner_mask.bool() & valid.bool()

    def _structural_valid(
        self,
        sequences: dict[str, th.Tensor],
        chunks:    int,
        num_envs:  int,
    ) -> th.Tensor:
        valid = sequences.get("valid")
        if valid is not None:
            return valid.bool()
        return th.ones(
            chunks * num_envs,
            self.sequence_length,
            dtype=th.bool,
            device=next(iter(sequences.values())).device,
        )

    def _sample_epoch(
        self,
        sequences:      dict[str, th.Tensor],
        clean:          th.Tensor,
        resetting:      th.Tensor,
        batch_sizes:    list[int],
        combined_valid: th.Tensor,
        structural_valid: th.Tensor,
    ):
        device = next(iter(sequences.values())).device
        clean = clean[th.randperm(len(clean), device=device)]

        if not len(resetting):
            left = 0
            for size in batch_sizes:
                selected = clean[left : left + size]
                yield self._build_batch(
                    sequences,
                    selected,
                    combined_valid,
                    structural_valid,
                    has_reset=False,
                )
                left += size
            return

        resetting = resetting[th.randperm(len(resetting), device=device)]
        indices = th.cat((resetting, clean))
        batches = []
        left = 0
        for size in batch_sizes:
            right = left + size
            batches.append((indices[left:right], left < len(resetting)))
            left = right

        # The batch count is small; keep this control-plane permutation on CPU
        # instead of synchronizing a CUDA tensor back to Python.
        for batch in th.randperm(len(batches), device="cpu").tolist():
            selected, has_reset = batches[batch]
            yield self._build_batch(
                sequences,
                selected,
                combined_valid,
                structural_valid,
                has_reset=has_reset,
            )

    def _batch_sizes(self, sequence_count: int) -> list[int]:
        """Cache the balanced optimizer-batch layout across epochs."""
        batch_count = (sequence_count + self.sequences_per_batch - 1) // (
            self.sequences_per_batch
        )
        small_batch, larger_batches = divmod(sequence_count, batch_count)
        return [
            small_batch + (batch < larger_batches)
            for batch in range(batch_count)
        ]

    @staticmethod
    def _build_batch(
        sequences:      dict[str, th.Tensor],
        selected:       th.Tensor,
        combined_valid: th.Tensor,
        structural_valid: th.Tensor,
        *,
        has_reset:      bool,
    ) -> SequenceBatch:
        state = sequences["policy_state"].index_select(0, selected)[:, 0]
        critic_state = sequences.get("critic_state")

        if critic_state is not None:
            critic_state = critic_state.index_select(0, selected)[:, 0]

        step_data = {
            # Gather directly into the time-major contiguous layout required by
            # cuDNN RNNs instead of transposing a batch-major gathered tensor.
            key: value.swapaxes(0, 1).index_select(1, selected)
            for key, value in sequences.items()
            if key not in ("policy_state", "critic_state")
        }
        steps = TensorBatch(step_data)
        valid = combined_valid.index_select(0, selected).swapaxes(0, 1)

        reset = None
        if has_reset:
            reset_valid = structural_valid.index_select(0, selected).swapaxes(0, 1)
            done = (steps["terminated"] | steps["truncated"]) & reset_valid
            reset = th.zeros_like(done)
            reset[1:] = done[:-1]

        return SequenceBatch(
            steps=steps,
            initial_state=state,
            reset=reset,
            valid=valid.bool(),
            initial_critic_state=critic_state,
        )
