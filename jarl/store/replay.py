import torch as th

from jarl.data.batch import TensorBatch


class ReplayBuffer:
    """Circular replay arranged as time-major vector-environment lanes."""

    def __init__(
        self,
        capacity: int,
        num_envs: int,
        storage_device: str | th.device = "cpu",
        sample_device: str | th.device | None = None,
    ) -> None:
        if capacity < 1 or num_envs < 1:
            raise ValueError("replay dimensions must be positive")
        self.capacity = capacity
        self.num_envs = num_envs
        self.storage_device = th.device(storage_device)
        self.sample_device = th.device(sample_device or storage_device)
        self.position = 0
        self.size = 0
        self._storage: dict[str, th.Tensor] | None = None

    def __len__(self) -> int:
        return self.size * self.num_envs

    def _initialize(self, record: dict[str, object]) -> None:
        self._storage = {}
        for key, value in record.items():
            tensor = th.as_tensor(value, device=self.storage_device)
            if len(tensor) != self.num_envs:
                raise ValueError(
                    f"field {key!r} has {len(tensor)} environments, "
                    f"expected {self.num_envs}"
                )
            self._storage[key] = th.empty(
                (self.capacity, *tensor.shape),
                dtype=tensor.dtype,
                device=self.storage_device,
            )

    def append(self, record: dict[str, object]) -> None:
        if self._storage is None:
            self._initialize(record)
        if record.keys() != self._storage.keys():
            raise KeyError("replay record fields changed after initialization")
        for key, value in record.items():
            tensor = th.as_tensor(value, device=self.storage_device)
            if tensor.shape != self._storage[key].shape[1:]:
                raise ValueError(f"field {key!r} changed shape")
            self._storage[key][self.position].copy_(tensor)
        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def ready(self, batch_size: int) -> bool:
        return len(self) >= batch_size

    def _physical_time(self, logical_time: th.Tensor) -> th.Tensor:
        oldest = self.position if self.size == self.capacity else 0
        return (logical_time + oldest) % self.capacity

    def sample(self, batch_size: int) -> TensorBatch:
        if not self.ready(batch_size) or self._storage is None:
            raise RuntimeError("replay does not contain enough transitions")
        logical_time = th.randint(self.size, (batch_size,))
        env = th.randint(self.num_envs, (batch_size,))
        physical = self._physical_time(logical_time)
        data = {
            key: value[physical.to(value.device), env.to(value.device)].to(
                self.sample_device
            )
            for key, value in self._storage.items()
        }
        data["replay_time"] = physical.to(self.sample_device)
        data["replay_env"] = env.to(self.sample_device)
        return TensorBatch(data)

    def sample_windows(self, batch_size: int, length: int) -> TensorBatch:
        if length < 1 or self.size < length or self._storage is None:
            raise RuntimeError("replay does not contain complete windows")
        done = self._storage["terminated"] | self._storage["truncated"]
        starts = []
        envs = []
        attempts = 0
        max_attempts = max(100, batch_size * 20)
        while len(starts) < batch_size and attempts < max_attempts:
            start = int(th.randint(self.size - length + 1, ()).item())
            env = int(th.randint(self.num_envs, ()).item())
            logical = th.arange(start, start + length)
            physical = self._physical_time(logical)
            if length == 1 or not done[physical[:-1], env].any():
                starts.append(start)
                envs.append(env)
            attempts += 1
        if len(starts) != batch_size:
            raise RuntimeError("could not sample enough episode-safe windows")

        start = th.tensor(starts)
        env = th.tensor(envs)
        logical = start[None, :] + th.arange(length)[:, None]
        physical = self._physical_time(logical)
        data = {
            key: value[physical.to(value.device), env.to(value.device)[None, :]].to(
                self.sample_device
            )
            for key, value in self._storage.items()
        }
        data["replay_time"] = physical.to(self.sample_device)
        data["replay_env"] = env.to(self.sample_device).expand(length, -1)
        return TensorBatch(data)
