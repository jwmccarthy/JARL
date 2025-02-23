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
        self.cnn.build().to(env.device)
        self.feats = len(self(env.obs_space.sample()))
        return self
    
    def forward(self, x: th.Tensor) -> th.Tensor:
        return self.cnn(x)