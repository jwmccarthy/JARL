import torch as th
import torch.nn as nn

from jarl.data.records import Evaluation, PolicyOutput


class ActorCritic(nn.Module):
    def __init__(
        self,
        actor,
        critic,
        shared_state: bool = False,
    ) -> None:
        super().__init__()
        self.actor = actor
        self.critic = critic
        self.shared_state = shared_state
        self.device = th.device("cpu")
        self.built = False

    def build(self, env):
        if self.shared_state:
            self._build_shared(env)
        else:
            self._build_independent(env)

        self.built = True
        return self

    def _build_shared(self, env) -> None:
        if self.actor.head is not self.critic.head:
            raise ValueError("shared state requires the same head instance")
        if self.actor.body is not self.critic.body:
            raise ValueError("shared state requires the same body instance")

        head = self.actor.head
        body = self.actor.body
        if not head.built:
            head.build(env)
        if not getattr(body, "built", False):
            body.build(head.feats)
        if not hasattr(body, "feats"):
            raise TypeError("shared body must expose its output feature count")

        self.actor.build_composed(env, body.feats)
        self.critic.build_composed(env, body.feats)

    def _build_independent(self, env) -> None:
        self.actor.build(env)
        self.critic.build(env)

        actor_recurrent = self.actor.initial_state(1) is not None
        critic_recurrent = hasattr(self.critic.body, "initial_state")
        if actor_recurrent or critic_recurrent:
            raise NotImplementedError(
                "independent recurrent actor-critic state is not yet supported"
            )

    def to(self, device, *args, **kwargs):
        self.device = th.device(device)
        self.actor.device = self.device
        self.critic.device = self.device
        return super().to(device, *args, **kwargs)

    def initial_state(self, batch_size: int) -> th.Tensor | None:
        self._require_built()
        if not self.shared_state:
            return None
        return self.actor.initial_state(batch_size)

    def act(
        self,
        observation: th.Tensor,
        state:       th.Tensor | None = None,
        *,
        deterministic: bool = False,
    ) -> PolicyOutput:
        if not self.shared_state:
            output = self.actor.act(observation, state, deterministic=deterministic)
            output.extras["value"] = self.critic.value(observation)
            return output

        features, next_state = self._shared_features(observation, state)
        output = self.actor.act_from_features(
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
            evaluation = self.actor.evaluate_actions(
                observation, action, state, reset=reset
            )
            evaluation.value = self.critic.evaluate_values(observation)
            return evaluation

        features, _ = self._shared_features(observation, state, reset)
        evaluation = self.actor.evaluate_from_features(
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
        return self.actor.body_features(observation, state, reset)

    def _require_built(self) -> None:
        if not self.built:
            raise RuntimeError("actor-critic must be built before use")


__all__ = ["ActorCritic"]
