import torch as th
import torch.nn as nn

from typing import Self

from jarl.envs.vec import TorchGymEnv
from jarl.modules.encoder.base import Encoder
    

class ImageEncoder(Encoder):

    def __init__(self, cnn: nn.Module) -> None:
        super().__init__()
        self.cnn = cnn

    def build(self, env: TorchGymEnv) -> Self:
        super().build(env)
        self.cnn.build().to(env.device)
        self.feats = len(self(env.obs_space.sample()))
        return self
    
    def forward(self, x: th.Tensor) -> th.Tensor:
        batch_dim = x.shape[:-3]
        x = x.view(-1, *x.shape[-3:])
        x = self.cnn(x / 255.0)
        return x.view(*batch_dim, -1)