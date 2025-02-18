import torch.nn as nn

from typing import Self
from abc import ABC, abstractmethod

from jarl.envs.gym import TorchGymEnv


class Encoder(nn.Module, ABC):

    feats: int

    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def build(self, env: TorchGymEnv) -> Self:
        ...