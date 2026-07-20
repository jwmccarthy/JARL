import torch as th


class TensorStorage:
    def __init__(
        self,
        capacity: int,
        num_envs: int,
        device: str | th.device,
    ) -> None:
        if capacity < 1 or num_envs < 1:
            raise ValueError("storage dimensions must be positive")

        self.capacity = capacity
        self.num_envs = num_envs
        self.device = th.device(device)
        self._storage: dict[str, th.Tensor] | None = None

    def _initialize(self, transition: dict[str, object]) -> None:
        self._storage = {}

        for key, value in transition.items():
            tensor = th.as_tensor(value, device=self.device)

            if len(tensor) != self.num_envs:
                raise ValueError(
                    f"field {key!r} has {len(tensor)} environments, "
                    f"expected {self.num_envs}"
                )

            self._storage[key] = th.empty(
                (self.capacity, *tensor.shape),
                dtype=tensor.dtype,
                device=self.device,
            )

    def _write(self, index: int, transition: dict[str, object]) -> None:
        if self._storage is None:
            self._initialize(transition)

        if transition.keys() != self._storage.keys():
            raise KeyError("transition fields changed after storage initialization")

        for key, value in transition.items():
            tensor = th.as_tensor(value, device=self.device)
            expected = self._storage[key].shape[1:]

            if tensor.shape != expected:
                raise ValueError(
                    f"field {key!r} has shape {tuple(tensor.shape)}, "
                    f"expected {tuple(expected)}"
                )

            self._storage[key][index].copy_(tensor)
