import torch as th
import torch.nn as nn

from typing import Self

from jarl.envs.gym import SyncGymEnv
from jarl.modules.encoder.core import Encoder


class SharedTrunk(nn.Module):

    def __init__(self, foot: Encoder, body: nn.Module) -> None:
        super().__init__()
        self.foot = foot
        self.body = body
        self.device = th.device("cpu")
        self.built = False
        self.feats: int | None = None

    def build(self, env: SyncGymEnv) -> Self:
        if not self.foot.built:
            self.foot.build(env)
        if not getattr(self.body, "built", False):
            self.body.build(self.foot.feats)

        self.feats = self.body.feats
        self.built = True

        return self

    def forward(
        self,
        observation: th.Tensor,
        state:       th.Tensor | None = None,
        reset:       th.Tensor | None = None,
    ) -> tuple[th.Tensor, th.Tensor | None]:
        features = self.foot(observation)

        if hasattr(self.body, "initial_state"):
            if (
                state is not None
                and state.dtype != features.dtype
                and th.is_autocast_enabled()
            ):
                state = state.to(features.dtype)
            return self.body(features, state, reset)
        
        if state is not None or reset is not None:
            raise ValueError("stateless trunk does not accept state")
        
        return self.body(features), None


__all__ = ["SharedTrunk"]
