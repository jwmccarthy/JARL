from dataclasses import dataclass
from typing import Protocol

import torch as th

from jarl.data.records import ActionDecision, EnvStep


@dataclass
class CaptureContext:
    obs: th.Tensor
    state_in: th.Tensor | None
    decision: ActionDecision
    env_step: EnvStep


class Capture(Protocol):
    produces: frozenset[str]

    def __call__(self, context: CaptureContext) -> dict[str, th.Tensor]:
        ...


class DecisionArtifact:
    def __init__(self, artifact: str, field: str) -> None:
        self.artifact = artifact
        self.field = field
        self.produces = frozenset({field})

    def __call__(self, context: CaptureContext) -> dict[str, th.Tensor]:
        try:
            value = context.decision.artifacts[self.artifact]
        except KeyError:
            raise KeyError(f"action decision has no {self.artifact!r} artifact") from None
        return {self.field: value}


class RecurrentStateCapture:
    produces = frozenset({"behavior_state"})

    def __call__(self, context: CaptureContext) -> dict[str, th.Tensor]:
        if context.state_in is None:
            raise ValueError("cannot capture an empty recurrent state")
        return {"behavior_state": context.state_in}


class BehaviorVersionCapture:
    produces = frozenset({"policy_version"})

    def __init__(self, behavior) -> None:
        self.behavior = behavior

    def __call__(self, context: CaptureContext) -> dict[str, th.Tensor]:
        boundary = th.as_tensor(
            context.env_step.terminated,
            device=context.obs.device,
        )
        return {
            "policy_version": th.full_like(
                boundary,
                self.behavior.version,
                dtype=th.long,
            )
        }


class ValueCapture:
    produces = frozenset(
        {"baseline_value", "baseline_next_value", "value_version"}
    )

    def __init__(self, estimator) -> None:
        self.estimator = estimator

    @th.no_grad()
    def __call__(self, context: CaptureContext) -> dict[str, th.Tensor]:
        next_obs = th.as_tensor(
            context.env_step.next_obs,
            device=context.obs.device,
        )
        return {
            "baseline_value": self.estimator.value(context.obs, context.state_in),
            "baseline_next_value": self.estimator.value(
                next_obs,
                context.decision.next_state,
            ),
            "value_version": th.full_like(
                th.as_tensor(
                    context.env_step.terminated,
                    device=context.obs.device,
                ),
                self.estimator.version,
                dtype=th.long,
            ),
        }


def validate_captures(captures: list[Capture] | tuple[Capture, ...]) -> None:
    produced: set[str] = set()
    for capture in captures:
        duplicate = produced & capture.produces
        if duplicate:
            raise ValueError(f"duplicate capture fields: {sorted(duplicate)}")
        produced.update(capture.produces)


def build_record(
    context: CaptureContext,
    captures: list[Capture] | tuple[Capture, ...],
) -> dict[str, th.Tensor]:
    record = {
        "obs": context.obs,
        "act": context.decision.action,
        "rew": context.env_step.reward,
        "next_obs": context.env_step.next_obs,
        "terminated": context.env_step.terminated,
        "truncated": context.env_step.truncated,
    }
    for capture in captures:
        fields = capture(context)
        duplicate = record.keys() & fields.keys()
        if duplicate:
            raise ValueError(f"capture would overwrite fields: {sorted(duplicate)}")
        record.update(fields)
    return record
