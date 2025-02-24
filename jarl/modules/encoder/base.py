import torch.nn as nn

from typing import Self
from abc import ABC, abstractmethod

from jarl.envs.vec import TorchGymEnv


class Encoder(nn.Module, ABC):

    feats: int
    built: bool = False

    def __init__(self) -> None:
        super().__init__()

    def build(self, env: TorchGymEnv) -> Self:
        self.built = True