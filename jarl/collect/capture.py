from dataclasses import dataclass
import torch as th

from jarl.data.records import EnvStep, PolicyOutput


@dataclass
class CaptureContext:
    observation: th.Tensor
    state: th.Tensor | None
    policy_output: PolicyOutput
    env_step: EnvStep


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


class ValueCapture:
    def __init__(self, estimator) -> None:
        self.estimator = estimator

    @th.no_grad()
    def __call__(self, context: CaptureContext) -> dict[str, th.Tensor]:
        next_obs = th.as_tensor(
            context.env_step.next_obs,
            device=context.observation.device,
        )
        baseline_value = context.policy_output.extras.get("value")
        learner_mask = context.policy_output.extras.get("learner_mask")
        if baseline_value is None:
            baseline_value = self.estimator.value(
                context.observation,
                context.state,
            )
        if learner_mask is None:
            baseline_next_value = self.estimator.value(
                next_obs,
                context.policy_output.next_state,
            )
        else:
            baseline_next_value = th.zeros_like(baseline_value)
            next_state = context.policy_output.next_state
            learner_state = None if next_state is None else next_state[learner_mask]
            baseline_next_value[learner_mask] = self.estimator.value(
                next_obs[learner_mask],
                learner_state,
            )
        return {
            "baseline_value":      baseline_value,
            "baseline_next_value": baseline_next_value,
        }


def build_record(
    context: CaptureContext,
    captures,
) -> dict[str, th.Tensor]:
    bootstrap = context.env_step.bootstrap
    if bootstrap is None:
        bootstrap = ~th.as_tensor(context.env_step.terminated, dtype=th.bool)

    record = {
        "observation": context.observation,
        "action":      context.policy_output.action,
        "reward":      context.env_step.reward,
        "next_obs":    context.env_step.next_obs,
        "terminated":  context.env_step.terminated,
        "truncated":   context.env_step.truncated,
        "bootstrap":   bootstrap,
    }

    for capture in captures:
        record.update(capture(context))

    return record
