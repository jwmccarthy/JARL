from collections.abc import Callable, Iterable
from dataclasses import dataclass

import torch as th

from jarl.data.batch import TensorBatch
from jarl.data.dataset import TensorDataset


@dataclass(frozen=True)
class ResetContext:
    reset_mask:          th.Tensor
    environment_indices: th.Tensor
    dataset_indices:     th.Tensor
    generator:           th.Generator


ResetTransform = Callable[[TensorBatch, ResetContext], TensorBatch]


class DatasetResetSampler:
    def __init__(
        self,
        dataset:     TensorDataset,
        *,
        transforms:  Iterable[ResetTransform] = (),
        probability: float = 1.0,
        seed:        int = 0,
    ) -> None:
        if not isinstance(dataset, TensorDataset):
            raise TypeError("dataset must be a TensorDataset")
        if not 0.0 <= probability <= 1.0:
            raise ValueError("reset probability must be between zero and one")

        self.dataset = dataset
        self.transforms = tuple(transforms)
        self.probability = probability
        if not all(callable(transform) for transform in self.transforms):
            raise TypeError("reset transforms must be callable")

        self._generator = th.Generator(device=dataset.device).manual_seed(seed)

    def __call__(self, reset_mask: th.Tensor) -> TensorBatch | None:
        if not isinstance(reset_mask, th.Tensor):
            raise TypeError("reset_mask must be a tensor")
        if reset_mask.dtype != th.bool:
            raise TypeError("reset_mask must have dtype bool")
        if reset_mask.ndim != 1:
            raise ValueError("reset_mask must be one-dimensional")
        if reset_mask.device != self.dataset.device:
            raise ValueError("reset_mask must be on the dataset device")

        environment_indices = reset_mask.nonzero(as_tuple=True)[0]
        count = environment_indices.numel()
        if not count:
            return None

        if self.probability == 0.0:
            return None
        if self.probability < 1.0:
            selected = th.rand(
                count,
                device=self.dataset.device,
                generator=self._generator,
            ) < self.probability
            environment_indices = environment_indices[selected]
            count = environment_indices.numel()
            if not count:
                return None

        dataset_indices = th.randint(
            len(self.dataset),
            (count,),
            device=self.dataset.device,
            generator=self._generator,
        )
        sample = self.dataset[dataset_indices]
        context = ResetContext(
            reset_mask=reset_mask,
            environment_indices=environment_indices,
            dataset_indices=dataset_indices,
            generator=self._generator,
        )

        for transform in self.transforms:
            sample = transform(sample, context)
            self._validate_sample(sample, count)

        return sample.with_fields(simulation_indices=environment_indices)

    def _validate_sample(
        self,
        sample: TensorBatch,
        count:  int,
    ) -> None:
        if not isinstance(sample, TensorBatch):
            raise TypeError("reset transforms must return a TensorBatch")
        if len(sample) != count:
            raise ValueError("reset transforms cannot change the sample count")
        if any(value.device != self.dataset.device for value in sample.values()):
            raise ValueError("reset transforms cannot change the sample device")
