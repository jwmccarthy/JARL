import torch as th
from torch.distributions import Distribution

from typing import Self

from jarl.data.records import Evaluation, PolicyOutput
from jarl.envs.gym import SyncGymEnv
from jarl.modules.operator import Critic
from jarl.modules.policy import Policy
from jarl.modules.trunk import SharedTrunk


class ActorCritic(Policy):

    def __init__(
        self,
        policy: Policy,
        critic: Critic,
    ) -> None:
        super().__init__(policy.foot, policy.body, policy.head)
        self.policy = policy
        self.critic = critic
        self.shared_state = policy.foot is critic.foot
        self.trunk = (
            SharedTrunk(policy.foot, policy.body)
            if self.shared_state
            else None
        )
        self.device = th.device("cpu")
        self.built = False

    def dist(self, observation: th.Tensor) -> Distribution:
        return self.policy.dist(observation)

    def action(self, observation: th.Tensor) -> th.Tensor:
        return self.policy.action(observation)

    def sample(self, observation: th.Tensor) -> th.Tensor:
        return self.policy.sample(observation)

    def build(self, env: SyncGymEnv) -> Self:
        if self.shared_state:
            self._build_shared(env)
        else:
            self._build_independent(env)

        self.built = True
        return self

    def _build_shared(self, env: SyncGymEnv) -> None:
        if self.policy.foot is not self.critic.foot:
            raise ValueError("shared state requires the same foot instance")
        if self.policy.body is not self.critic.body:
            raise ValueError("shared state requires the same body instance")

        if self.trunk is None:
            raise RuntimeError("shared actor-critic is missing its trunk")
        self.trunk.build(env)
        if self.trunk.feats is None:
            raise TypeError("shared trunk must expose its output feature count")

        self.policy._build_shared_head(env, self.trunk.feats)
        self.critic._build_shared_head(self.trunk.feats)

    def _build_independent(self, env: SyncGymEnv) -> None:
        self.policy.build(env)
        self.critic.build(env)

        policy_recurrent = self.policy.initial_state(1) is not None
        critic_recurrent = self.critic.initial_state(1) is not None
        
        if policy_recurrent or critic_recurrent:
            raise NotImplementedError(
                "independent recurrent actor-critic state is not yet supported"
            )

    def to(self, device: str, *args, **kwargs) -> Self:
        self.device = th.device(device)
        self.policy.device = self.device
        self.critic.device = self.device
        return super().to(device, *args, **kwargs)

    def initial_state(self, batch_size: int) -> th.Tensor | None:
        self._require_built()
        if not self.shared_state:
            return None
        return self.policy.initial_state(batch_size)

    def act(
        self,
        observation:   th.Tensor,
        state:         th.Tensor | None = None,
        *,
        deterministic: bool = False,
    ) -> PolicyOutput:
        if not self.shared_state:
            output = self.policy.act(observation, state, deterministic=deterministic)
            output.extras["value"] = self.critic.value(observation)
            return output

        features, next_state = self._shared_features(observation, state)
        output = self.policy.act_from_features(
            features,
            observation,
            deterministic=deterministic,
        )
        output.next_state = next_state
        output.extras["value"] = self.critic.value_from_features(features)

        return output

    def evaluate_actions(
        self,
        observation: th.Tensor,
        action:      th.Tensor,
        state:       th.Tensor | None = None,
        *,
        reset:       th.Tensor | None = None,
    ) -> Evaluation:
        if not self.shared_state:
            evaluation = self.policy.evaluate_actions(
                observation, action, state, reset=reset
            )
            evaluation.value = self.critic.evaluate_values(observation)
            return evaluation

        features, _ = self._shared_features(observation, state, reset)
        evaluation = self.policy.evaluate_from_features(
            features,
            observation,
            action,
        )
        evaluation.value = self.critic.value_from_features(features)

        return evaluation

    def value(
        self,
        observation: th.Tensor,
        state:       th.Tensor | None = None,
    ) -> th.Tensor:
        if not self.shared_state:
            return self.critic.value(observation, state)
        features, _ = self._shared_features(observation, state)
        return self.critic.value_from_features(features)

    def evaluate_values(
        self,
        observation: th.Tensor,
        state:       th.Tensor | None = None,
        *,
        reset:       th.Tensor | None = None,
    ) -> th.Tensor:
        if not self.shared_state:
            return self.critic.evaluate_values(observation, state, reset=reset)
        features, _ = self._shared_features(observation, state, reset)
        return self.critic.value_from_features(features)

    def _shared_features(
        self,
        observation: th.Tensor,
        state:       th.Tensor | None,
        reset:       th.Tensor | None = None,
    ) -> tuple[th.Tensor, th.Tensor | None]:
        if self.trunk is None:
            raise RuntimeError("actor-critic has no shared trunk")
        return self.trunk(observation, state, reset)

    def _require_built(self) -> None:
        if not self.built:
            raise RuntimeError("actor-critic must be built before use")


__all__ = ["ActorCritic"]
