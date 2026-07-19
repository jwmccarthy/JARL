from dataclasses import dataclass, field

from jarl.store.rollout import Rollout
from jarl.transform.base import PrepareContext, apply_transforms


@dataclass
class LearningWorkspace:
    experience: dict[str, object]
    metrics:    dict[str, dict[str, float]] = field(default_factory=dict)

    def require(self, name: str):
        try:
            return self.experience[name]
        except KeyError:
            raise RuntimeError(f"missing learning experience {name!r}") from None

    def publish(self, name: str, value: object) -> None:
        if name in self.experience:
            raise RuntimeError(f"learning experience {name!r} already exists")
        self.experience[name] = value

    def add_metrics(self, section: str, values: dict[str, float]) -> None:
        if section in self.metrics:
            raise RuntimeError(f"metrics section {section!r} already exists")
        self.metrics[section] = values


class LearningProgram:
    def __init__(self, stages) -> None:
        self.stages = tuple(stages)

    def update(self, experience) -> dict[str, dict[str, float]]:
        workspace = LearningWorkspace({"experience": experience})

        for stage in self.stages:
            stage.run(workspace)

        return workspace.metrics


class TransformRollout:
    def __init__(self, rollout: str, output: str, transforms) -> None:
        self.rollout = rollout
        self.output = output
        self.transforms = tuple(transforms)

    def run(self, workspace: LearningWorkspace) -> None:
        rollout = workspace.require(self.rollout)
        if not isinstance(rollout, Rollout):
            raise TypeError("TransformRollout requires a Rollout")

        steps = apply_transforms(
            rollout.steps,
            self.transforms,
            PrepareContext(rollout),
        )
        workspace.publish(self.output, rollout.with_steps(steps))


class OptimizePPO:
    def __init__(self, rollout: str, optimizer, section: str = "PPO") -> None:
        self.rollout = rollout
        self.optimizer = optimizer
        self.section = section

    def run(self, workspace: LearningWorkspace) -> None:
        rollout = workspace.require(self.rollout)
        workspace.add_metrics(
            self.section,
            self.optimizer.update(rollout.steps),
        )
