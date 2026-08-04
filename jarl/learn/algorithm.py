from jarl.store.rollout import Rollout
from jarl.transform.base import PrepareContext, apply_transforms


class Algorithm:
    def __init__(self, *stages) -> None:
        self.stages = stages

    def update(self, experience) -> dict[str, dict[str, float]]:
        metrics = {}

        for stage in self.stages:
            experience, stage_metrics = stage.run(experience)
            metrics.update(stage_metrics)

        return metrics

    def set_progress_callback(self, callback) -> None:
        for stage in self.stages:
            stage.set_progress_callback(callback)


class TransformRollout:
    def __init__(
        self,
        *transforms,
        report_fields: tuple[str, ...] = (),
        section: str = "Transform",
    ) -> None:
        self.transforms = transforms
        self.report_fields = report_fields
        self.section = section

    def set_progress_callback(self, callback) -> None:
        return

    def run(self, rollout):
        if not isinstance(rollout, Rollout):
            raise TypeError("TransformRollout requires a Rollout")

        steps = apply_transforms(
            rollout.steps,
            self.transforms,
            PrepareContext(rollout),
        )
        metrics = {}

        if self.report_fields:
            valid = steps.get("learner_mask")
            values = {}

            for field in self.report_fields:
                value = steps[field]
                if valid is not None:
                    value = value[valid.bool()]
                values[field] = value.mean().item()

            metrics[self.section] = values
            
        return rollout.with_steps(steps), metrics
