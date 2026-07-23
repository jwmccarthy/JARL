from dataclasses import dataclass

from jarl.data.batch import TensorBatch
from jarl.store.rollout import Rollout


@dataclass(frozen=True)
class PrepareContext:
    rollout: Rollout | None = None


def apply_transforms(
    batch:      TensorBatch,
    transforms,
    context:    PrepareContext | None = None,
) -> TensorBatch:
    context = context or PrepareContext()

    for transform in transforms:
        batch = transform(batch, context)

    return batch
