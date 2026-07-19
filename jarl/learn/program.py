from dataclasses import dataclass, field
from typing import Protocol

from jarl.store.rollout import Rollout
from jarl.transform.base import PrepareContext, apply_transforms


@dataclass
class LearningWorkspace:
    artifacts: dict[str, object]
    metrics: dict[str, dict[str, float]] = field(default_factory=dict)

    def require(self, name: str):
        try:
            return self.artifacts[name]
        except KeyError:
            raise RuntimeError(f"missing learning artifact {name!r}") from None

    def publish(self, name: str, value: object) -> None:
        if name in self.artifacts:
            raise RuntimeError(f"learning artifact {name!r} already exists")
        self.artifacts[name] = value

    def add_metrics(self, section: str, values: dict[str, float]) -> None:
        if section in self.metrics:
            raise RuntimeError(f"metrics section {section!r} already exists")
        self.metrics[section] = values


class LearningStage(Protocol):
    requires: frozenset[str]
    produces: frozenset[str]

    def run(self, workspace: LearningWorkspace) -> None:
        ...


class LearningProgram:
    def __init__(self, stages) -> None:
        self.stages = tuple(stages)
        available = {"source"}
        for stage in self.stages:
            missing = stage.requires - available
            if missing:
                raise ValueError(
                    f"{type(stage).__name__} requires unavailable artifacts "
                    f"{sorted(missing)}"
                )
            duplicate = stage.produces & available
            if duplicate:
                raise ValueError(
                    f"{type(stage).__name__} duplicates artifacts "
                    f"{sorted(duplicate)}"
                )
            available.update(stage.produces)

    def update(self, source) -> dict[str, dict[str, float]]:
        workspace = LearningWorkspace({"source": source})
        for stage in self.stages:
            stage.run(workspace)
        return workspace.metrics


class TransformArtifact:
    def __init__(self, source: str, output: str, transforms) -> None:
        self.source = source
        self.output = output
        self.transforms = tuple(transforms)
        self.requires = frozenset({source})
        self.produces = frozenset({output})

    def run(self, workspace: LearningWorkspace) -> None:
        source = workspace.require(self.source)
        if not isinstance(source, Rollout):
            raise TypeError("TransformArtifact currently requires a Rollout")
        steps = apply_transforms(
            source.steps,
            self.transforms,
            PrepareContext(source),
        )
        workspace.publish(self.output, source.with_steps(steps))


class OptimizePPO:
    def __init__(self, source: str, optimizer, section: str = "PPO") -> None:
        self.source = source
        self.optimizer = optimizer
        self.section = section
        self.requires = frozenset({source})
        self.produces = frozenset()

    def run(self, workspace: LearningWorkspace) -> None:
        rollout = workspace.require(self.source)
        workspace.add_metrics(
            self.section,
            self.optimizer.update(rollout.steps),
        )
