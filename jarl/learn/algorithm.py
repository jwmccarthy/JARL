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


class TransformRollout:
    def __init__(self, *transforms) -> None:
        self.transforms = transforms

    def run(self, rollout):
        if not isinstance(rollout, Rollout):
            raise TypeError("TransformRollout requires a Rollout")

        steps = apply_transforms(
            rollout.steps,
            self.transforms,
            PrepareContext(rollout),
        )
        return rollout.with_steps(steps), {}
