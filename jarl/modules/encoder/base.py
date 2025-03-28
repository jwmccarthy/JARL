import torch.nn as nn

from typing import Self
from abc import ABC, abstractmethod

from jarl.envs.gym import SyncGymEnv


class Encoder(nn.Module, ABC):

    feats: int
    built: bool = False

    def __init__(self) -> None:
        super().__init__()

    def build(self, env: SyncGymEnv) -> Self:
        self.built = True