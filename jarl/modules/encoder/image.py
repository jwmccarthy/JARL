import torch as th
import torch.nn as nn

from typing import Self

from jarl.envs.gym import SyncEnv
from jarl.modules.encoder.base import Encoder
    

class ImageEncoder(Encoder):

    def __init__(self, cnn: nn.Module) -> None:
        super().__init__()
        self.cnn = cnn

    def build(self, env: SyncEnv) -> Self:
        super().build(env)
        self.cnn.build(env.obs_space.shape[-3]).to(env.device)
        self.feats = len(self(env.obs_space.sample()))
        return self
    
    def forward(self, x: th.Tensor) -> th.Tensor:
        batch_dim = x.shape[:-3]
        x = x.view(-1, *x.shape[-3:])
        return self.cnn(x).view(*batch_dim, -1)