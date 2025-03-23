import torch as th
import torch.nn as nn

from typing import Self

from jarl.envs.gym import SyncGymEnv
from jarl.modules.encoder.base import Encoder
    

class ImageEncoder(Encoder):

    def __init__(self, cnn: nn.Module) -> None:
        super().__init__()
        self.cnn = cnn

    def build(self, env: SyncGymEnv) -> Self:
        super().build(env)
        self.obs_dim = env.obs_space.shape
        self.cnn.build(self.obs_dim[-3])
        self.feats = self(env.obs_space.sample()).shape[-1]
        return self
    
    def forward(self, x: th.Tensor) -> th.Tensor:
        batch_dim = x.shape[:-3]
        x = x.view(-1, *x.shape[-3:]) / 255.0
        return self.cnn(x).view(*batch_dim, -1)