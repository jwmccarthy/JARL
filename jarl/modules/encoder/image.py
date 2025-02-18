import torch as th
import torch.nn as nn
from torchvision.transforms import v2

from typing import Self

from jarl.envs.gym import TorchGymEnv
from jarl.modules.encoder.base import Encoder
    

class ImageEncoder(Encoder):

    def __init__(
        self, 
        cnn: nn.Module,
        transform: v2.Compose = None
    ) -> None:
        super().__init__()
        self.cnn = cnn
        self.transform = transform

    def build(self, env: TorchGymEnv) -> Self:
        self.cnn.build().to(env.device)
        self.feats = len(self(env.obs_space.sample()))
        return self
    
    def forward(self, x: th.Tensor) -> th.Tensor:
        x = th.movedim(x, -1, -3) / 255.0
        if self.transform:
            x = self.transform(x)
        return self.cnn(x)