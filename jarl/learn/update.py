from dataclasses import dataclass

import torch as th

from jarl.data.batch import TensorBatch
from jarl.store.rollout import Rollout
from jarl.transform.base import PrepareContext, apply_transforms


@dataclass(frozen=True)
class LossOutput:
    loss: th.Tensor
    metrics: dict[str, float]


class Update:
    def __init__(
        self,
        transforms,
        sampler,
        loss,
        optimizer_step,
        section: str = "Update",
    ) -> None:
        self.transforms = tuple(transforms)
        self.sampler = sampler
        self.loss = loss
        self.optimizer_step = optimizer_step
        self.section = section

    def update(self, experience: Rollout | TensorBatch) -> dict[str, dict[str, float]]:
        if isinstance(experience, Rollout):
            batch = experience.steps
            context = PrepareContext(experience)
        elif isinstance(experience, TensorBatch):
            batch = experience
            context = PrepareContext()
        else:
            raise TypeError("Update requires a Rollout or TensorBatch")

        prepared = apply_transforms(batch, self.transforms, context)
        validate = getattr(self.loss, "validate", None)
        if validate is not None:
            validate(prepared)

        totals: dict[str, float] = {}
        count = 0
        for sample in self.sampler(prepared):
            output = self.loss(sample)
            if isinstance(output, th.Tensor):
                output = LossOutput(output, {"loss": output.item()})
            elif not isinstance(output, LossOutput):
                raise TypeError("loss must return a tensor or LossOutput")

            self.optimizer_step(output.loss)
            for key, value in output.metrics.items():
                totals[key] = totals.get(key, 0.0) + value
            count += 1

        if count == 0:
            raise RuntimeError("sampler produced no minibatches")

        self.optimizer_step.advance_scheduler()
        after_update = getattr(self.loss, "after_update", None)
        if after_update is not None:
            after_update()

        metrics = {key: value / count for key, value in totals.items()}
        return {self.section: metrics}
