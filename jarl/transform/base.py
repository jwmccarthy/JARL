from dataclasses import dataclass
from typing import Protocol

from jarl.data.batch import TensorBatch
from jarl.store.rollout import Rollout


@dataclass(frozen=True)
class PrepareContext:
    rollout: Rollout | None = None


class Transform(Protocol):
    requires: frozenset[str]
    produces: frozenset[str]
    replaces: frozenset[str]

    def __call__(
        self, batch: TensorBatch, context: PrepareContext
    ) -> TensorBatch:
        ...


def apply_transforms(
    batch: TensorBatch,
    transforms: list[Transform] | tuple[Transform, ...],
    context: PrepareContext | None = None,
) -> TensorBatch:
    context = context or PrepareContext()
    available = set(batch)
    for transform in transforms:
        missing = transform.requires - available
        if missing:
            raise ValueError(
                f"{type(transform).__name__} requires missing fields "
                f"{sorted(missing)}"
            )
        unexpected = transform.produces & available - transform.replaces
        if unexpected:
            raise ValueError(
                f"{type(transform).__name__} would overwrite fields "
                f"{sorted(unexpected)}"
            )
        batch = transform(batch, context)
        available.update(transform.produces)
    return batch
