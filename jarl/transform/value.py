import torch as th

from jarl.data.batch import TensorBatch
from jarl.transform.base import PrepareContext


class MaterializeValues:
    def __init__(self, estimator) -> None:
        self.estimator = estimator

    @th.no_grad()
    def __call__(
        self, batch: TensorBatch, context: PrepareContext
    ) -> TensorBatch:
        if "policy_state" in batch:
            raise ValueError(
                "recurrent values must be captured until recurrent "
                "materialization has an explicit state-unroll contract"
            )

        value = self.estimator.value(batch["observation"])
        next_value = self.estimator.value(batch["next_obs"])
        return batch.with_fields(
            baseline_value=value,
            baseline_next_value=next_value,
        )
