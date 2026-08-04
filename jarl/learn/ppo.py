from dataclasses import dataclass

import torch as th

from jarl.data.batch import TensorBatch
from jarl.learn.update import LossOutput
from jarl.modules.actor_critic import ActorCritic
from jarl.modules.operator import Critic
from jarl.modules.policy import Policy
from jarl.sample.rollout import SequenceBatch


@dataclass(frozen=True)
class PPOConfig:
    clip:                float = 0.2
    value_clip:          float | None = 0.2
    value_coef:          float = 0.5
    entropy_coef:        float = 0.01
    normalize_advantage: bool = True
    bf16:                bool = False
    action_names:        tuple[str, ...] | None = None


class PPOLoss:
    
    def __init__(
        self,
        policy: Policy,
        critic: Critic | None = None,
        config: PPOConfig = PPOConfig(),
    ) -> None:
        if isinstance(policy, ActorCritic) != (critic is None):
            raise ValueError(
                "ActorCritic requires no critic; standalone policies require one"
            )

        self.policy = policy
        self.critic = critic
        self.config = config

    def after_update(self) -> None:
        return

    def __call__(self, sample: TensorBatch | SequenceBatch) -> LossOutput:
        batch, state, critic_state, reset, valid = self._unpack_sample(sample)

        with th.autocast(
            device_type=batch.device.type,
            dtype=th.bfloat16,
            enabled=self.config.bf16 and batch.device.type == "cuda",
        ):
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
        metrics.update(self._factor_metrics(evaluation, batch, valid))

        return LossOutput(loss, metrics)

    def _evaluate(self, batch, state, critic_state, reset):
        observation = batch["observation"]
        action = batch["action"]

        evaluation = self.policy.evaluate_actions(
            observation,
            action,
            state,
            reset=reset,
        )
        if (value := evaluation.value) is None:
            value = self.critic.evaluate_values(
                observation,
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
        ratio: th.Tensor,
    ) -> th.Tensor:
        clipped_ratio = ratio.clamp(
            1 - self.config.clip,
            1 + self.config.clip,
        )
        return -th.minimum(
            advantage * ratio,
            advantage * clipped_ratio,
        ).mean()

    def _value_loss(
        self,
        predicted_value: th.Tensor,
        batch: TensorBatch,
        valid: th.Tensor,
    ) -> th.Tensor:
        target = batch["returns"][valid]
        loss = (predicted_value - target).pow(2)

        if self.config.value_clip is None:
            return 0.5 * loss.mean()

        old_value = batch["baseline_value"][valid]
        clipped_value = old_value + (predicted_value - old_value).clamp(
            -self.config.value_clip,
            self.config.value_clip,
        )
        return 0.5 * th.maximum(
            loss,
            (clipped_value - target).pow(2),
        ).mean()

    def _factor_metrics(
        self,
        evaluation,
        batch: TensorBatch,
        valid: th.Tensor,
    ) -> dict[str, th.Tensor]:
        factor_entropy = evaluation.extras.get("factor_entropy")
        if factor_entropy is None:
            return {}

        factor_count = factor_entropy.shape[-1]
        names = self.config.action_names
        if names is None:
            names = tuple(str(index) for index in range(factor_count))
        elif len(names) != factor_count:
            raise ValueError("action names do not match policy action factors")

        factor_entropy = factor_entropy[valid]
        action = batch["action"].reshape(*valid.shape, -1)[valid]

        metrics = {}
        for index, name in enumerate(names):
            metrics[f"entropy_{name}"] = factor_entropy[:, index].mean()
            
            for value in range(self.policy.sizes[index]):
                metrics[f"action_{name}_{value}_rate"] = (
                    action[:, index].eq(value).float().mean()
                )

        return metrics
