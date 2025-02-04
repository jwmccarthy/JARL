import torch as th
import torch.nn as nn

from typing import Self

from jarl.envs.gym import TorchGymEnv
from jarl.modules.encoder import Encoder
from jarl.modules.composite import CompositeNet


class Discriminator(CompositeNet):

    def __init__(
        self,
        head: Encoder,
        body: nn.Module
    ) -> None:
        super().__init__(head, body)

    def build(self, env: TorchGymEnv) -> Self:
        self.head = self.head.build(env.obs_space)
        self.body.build(self.head.feats * 2, 1)
        return self

    def forward(
        self, 
        obs: th.Tensor, 
        next_obs: th.Tensor
    ) -> th.Tensor:
        obs_feats = self.head(obs)
        next_obs_feats = self.head(next_obs)
        feat_pair = th.cat([obs_feats, next_obs_feats], dim=1)
        return th.sigmoid(self.body(feat_pair)).squeeze()