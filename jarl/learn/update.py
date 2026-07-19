from dataclasses import dataclass

import torch as th

from jarl.data.batch import TensorBatch
from jarl.store.rollout import Rollout
from jarl.transform.base import PrepareContext, apply_transforms


@dataclass(frozen=True)
class LossOutput:
    loss:    th.Tensor
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

    def run(self, experience):
        return experience, self.update(experience)

    def update(self, experience: Rollout | TensorBatch) -> dict[str, dict[str, float]]:
        if isinstance(experience, Rollout):
            batch = experience.steps
            context = PrepareContext(experience)
        elif isinstance(experience, TensorBatch):
            batch = experience
            context = PrepareContext()
        else:
            raise TypeError("Update requires a Rollout or TensorBatch")

        prepared_batch = apply_transforms(batch, self.transforms, context)

        metric_totals: dict[str, float] = {}
        minibatch_count = 0

        for sample in self.sampler(prepared_batch):
            loss_output = self.loss(sample)

            if isinstance(loss_output, th.Tensor):
                loss_output = LossOutput(
                    loss_output,
                    {"loss": loss_output.item()},
                )
            elif not isinstance(loss_output, LossOutput):
                raise TypeError("loss must return a tensor or LossOutput")

            self.optimizer_step(loss_output.loss)

            for metric_name, metric_value in loss_output.metrics.items():
                metric_totals[metric_name] = (
                    metric_totals.get(metric_name, 0.0) + metric_value
                )

            minibatch_count += 1

        if minibatch_count == 0:
            raise RuntimeError("sampler produced no minibatches")

        self.optimizer_step.advance_scheduler()
        after_update = getattr(self.loss, "after_update", None)
        if after_update is not None:
            after_update()

        metrics = {
            metric_name: metric_value / minibatch_count
            for metric_name, metric_value in metric_totals.items()
        }

        return {self.section: metrics}
