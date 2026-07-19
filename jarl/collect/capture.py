from dataclasses import dataclass
import torch as th

from jarl.data.records import EnvStep, PolicyOutput


@dataclass
class CaptureContext:
    observation:   th.Tensor
    state:         th.Tensor | None
    policy_output: PolicyOutput
    env_step:      EnvStep


class LogProbCapture:
    def __call__(self, context: CaptureContext) -> dict[str, th.Tensor]:
        if context.policy_output.log_prob is None:
            raise ValueError("policy did not produce an action log probability")
        return {"old_log_prob": context.policy_output.log_prob}


class RecurrentStateCapture:
    def __call__(self, context: CaptureContext) -> dict[str, th.Tensor]:
        if context.state is None:
            raise ValueError("cannot capture an empty recurrent state")
        return {"policy_state": context.state}


class PolicyVersionCapture:
    def __init__(self, policy) -> None:
        self.policy = policy

    def __call__(self, context: CaptureContext) -> dict[str, th.Tensor]:
        boundary = th.as_tensor(
            context.env_step.terminated,
            device=context.observation.device,
        )
        return {
            "policy_version": th.full_like(
                boundary,
                self.policy.version,
                dtype=th.long,
            )
        }


class ValueCapture:
    def __init__(self, estimator) -> None:
        self.estimator = estimator

    @th.no_grad()
    def __call__(self, context: CaptureContext) -> dict[str, th.Tensor]:
        next_obs = th.as_tensor(
            context.env_step.next_obs,
            device=context.observation.device,
        )
        return {
            "baseline_value": self.estimator.value(
                context.observation,
                context.state,
            ),
            "baseline_next_value": self.estimator.value(
                next_obs,
                context.policy_output.next_state,
            ),
            "value_version": th.full_like(
                th.as_tensor(
                    context.env_step.terminated,
                    device=context.observation.device,
                ),
                self.estimator.version,
                dtype=th.long,
            ),
        }


def build_record(
    context: CaptureContext,
    captures,
) -> dict[str, th.Tensor]:
    record = {
        "observation": context.observation,
        "action": context.policy_output.action,
        "reward": context.env_step.reward,
        "next_obs": context.env_step.next_obs,
        "terminated": context.env_step.terminated,
        "truncated": context.env_step.truncated,
    }

    for capture in captures:
        record.update(capture(context))

    return record
