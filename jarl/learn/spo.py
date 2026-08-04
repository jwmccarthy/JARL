from dataclasses import dataclass

import torch as th

from jarl.data.batch import TensorBatch
from jarl.learn.update import LossOutput
from jarl.modules.actor_critic import ActorCritic
from jarl.modules.operator import Critic
from jarl.modules.policy import Policy
from jarl.sample.rollout import SequenceBatch


@dataclass(frozen=True)
class SPOConfig:
    ratio_epsilon:       float = 0.2
    value_clip:          float | None = 0.2
    value_coef:          float = 0.5
    entropy_coef:        float = 0.01
    normalize_advantage: bool = True


class SPOLoss:
    
    def __init__(
        self,
        policy: Policy,
        critic: Critic | None = None,
        config: SPOConfig = SPOConfig(),
    ) -> None:
        if isinstance(policy, ActorCritic) != (critic is None):
            raise ValueError(
                "ActorCritic requires no critic; standalone policies require one"
            )

        self.policy = policy
        self.critic = critic
        self.config = config
        if config.ratio_epsilon <= 0:
            raise ValueError("ratio epsilon must be positive")

    def after_update(self) -> None:
        return

    def __call__(self, sample: TensorBatch | SequenceBatch) -> LossOutput:
        batch, state, critic_state, reset, valid = self._unpack_sample(sample)

        evaluation, value = self._evaluate(
            batch,
            state,
            critic_state,
            reset,
        )

        log_prob = evaluation.log_prob.float()
        entropy = evaluation.entropy.float()
        value = value.float()

        advantage = batch["advantage"][valid]
        if self.config.normalize_advantage:
            advantage = (advantage - advantage.mean()) / (
                advantage.std(unbiased=False) + 1e-8
            )
        log_ratio = log_prob[valid] - batch["old_log_prob"][valid]
        ratio = log_ratio.exp()

        policy_loss = self._policy_loss(advantage, ratio)
        value_loss = self._value_loss(value[valid], batch, valid)
        entropy = entropy[valid].mean()
        loss = (
            policy_loss
            + self.config.value_coef * value_loss
            - self.config.entropy_coef * entropy
        )

        metrics = {
            "policy_loss": policy_loss,
            "critic_loss": value_loss,
            "entropy": entropy,
            "approx_kl": ((ratio - 1) - log_ratio).mean().detach(),
        }

        return LossOutput(loss, metrics)

    def _evaluate(self, batch, state, critic_state, reset):
        evaluation = self.policy.evaluate_actions(
            batch["observation"],
            batch["action"],
            state,
            reset=reset,
        )
        if (value := evaluation.value) is None:
            value = self.critic.evaluate_values(
                batch["observation"],
                critic_state,
                reset=reset,
            )
        return evaluation, value

    @staticmethod
    def _unpack_sample(sample: TensorBatch | SequenceBatch):
        if not isinstance(sample, SequenceBatch):
            valid = th.ones_like(sample["advantage"], dtype=th.bool)
            return sample, None, None, None, valid

        critic_state = sample.initial_critic_state
        if critic_state is None:
            critic_state = sample.initial_state

        return (
            sample.steps,
            sample.initial_state,
            critic_state,
            sample.reset,
            sample.valid,
        )

    def _policy_loss(
        self,
        advantage: th.Tensor,
        ratio:     th.Tensor,
    ) -> th.Tensor:
        return -(
            advantage * ratio
            - advantage.abs()
            / (2 * self.config.ratio_epsilon)
            * (ratio - 1).square()
        ).mean()

    def _value_loss(
        self,
        predicted_value: th.Tensor,
        batch: TensorBatch,
        valid: th.Tensor,
    ) -> th.Tensor:
        target = batch["returns"][valid]
        value_loss = (predicted_value - target).pow(2)

        if self.config.value_clip is not None:
            old_value = batch["baseline_value"][valid]
            clipped = old_value + (predicted_value - old_value).clamp(
                -self.config.value_clip,
                self.config.value_clip,
            )
            value_loss = th.maximum(value_loss, (clipped - target).pow(2))

        return 0.5 * value_loss.mean()
