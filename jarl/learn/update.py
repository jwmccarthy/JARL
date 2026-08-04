import torch as th

from collections.abc import Generator
from contextlib import contextmanager
from dataclasses import dataclass

from jarl.data.batch import TensorBatch
from jarl.store.rollout import Rollout
from jarl.transform.base import PrepareContext, apply_transforms


MetricValue = float | th.Tensor
Experience = Rollout | TensorBatch


@dataclass(frozen=True)
class LossOutput:
    loss:    th.Tensor
    metrics: dict[str, MetricValue]


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
        self._progress_callback = None

    def set_progress_callback(self, callback) -> None:
        self._progress_callback = callback

        self.sampler.set_epoch_callback(self._epoch_finished)

    def run(self, experience: Experience):
        return experience, self.update(experience)

    def update(
        self,
        experience: Experience,
    ) -> dict[str, dict[str, float]]:
        prepared_batch = self._prepare_batch(experience)

        with self._track_progress():
            metric_totals, minibatch_count = self._process_minibatches(
                prepared_batch
            )

        if minibatch_count == 0:
            raise RuntimeError("sampler produced no minibatches")

        self._finalize_update()

        return {
            self.section: self._average_metrics(
                metric_totals,
                minibatch_count,
            )
        }

    def _prepare_batch(self, experience: Experience) -> TensorBatch:
        batch, context = self._unpack_experience(experience)
        return apply_transforms(batch, self.transforms, context)

    @staticmethod
    def _unpack_experience(
        experience: Experience,
    ) -> tuple[TensorBatch, PrepareContext]:
        if isinstance(experience, Rollout):
            return experience.steps, PrepareContext(experience)

        if isinstance(experience, TensorBatch):
            return experience, PrepareContext()

        raise TypeError(
            "Update requires a Rollout or TensorBatch, "
            f"got {type(experience).__name__}"
        )

    def _process_minibatches(
        self,
        batch: TensorBatch,
    ) -> tuple[dict[str, MetricValue], int]:
        metric_totals: dict[str, MetricValue] = {}
        minibatch_count = 0

        for sample in self.sampler(batch):
            output = self._normalize_loss_output(self.loss(sample))

            self.optimizer_step(output.loss)
            self._accumulate_metrics(metric_totals, output.metrics)

            minibatch_count += 1

        return metric_totals, minibatch_count

    @staticmethod
    def _normalize_loss_output(output) -> LossOutput:
        if isinstance(output, th.Tensor):
            return LossOutput(
                loss=output,
                metrics={"loss": output},
            )

        if isinstance(output, LossOutput):
            return output

        raise TypeError(
            "loss must return a torch.Tensor or LossOutput, "
            f"got {type(output).__name__}"
        )

    @staticmethod
    def _accumulate_metrics(
        totals:  dict[str, MetricValue],
        metrics: dict[str, MetricValue],
    ) -> None:
        for name, value in metrics.items():
            detached_value = (
                value.detach()
                if isinstance(value, th.Tensor)
                else value
            )
            totals[name] = totals.get(name, 0.0) + detached_value

    @staticmethod
    def _average_metrics(
        totals: dict[str, MetricValue],
        count:  int,
    ) -> dict[str, float]:
        return {
            name: Update._to_float(total / count)
            for name, total in totals.items()
        }

    @staticmethod
    def _to_float(value: MetricValue) -> float:
        if isinstance(value, th.Tensor):
            return float(value.item())

        return float(value)

    def _finalize_update(self) -> None:
        self.optimizer_step.advance_scheduler()

        self.loss.after_update()

    @contextmanager
    def _track_progress(self) -> Generator[None, None, None]:
        callback = self._progress_callback

        if callback is None:
            yield
            return

        callback.start(self.sampler.epochs, self.section)
        try:
            yield
        finally:
            callback.finish()

    def _epoch_finished(self) -> None:
        callback = self._progress_callback
        if callback is not None:
            callback.epoch_finished()
