import torch as th

import pickle
from typing import Set

from jarl.data.core import MultiTensor
from jarl.train.modify.base import DataModifier


class GAIFODemos(DataModifier):

    def __init__(
        self,
        path: str,
        num_samples: int = 2048
    ) -> None:
        with open(path, "rb") as f:
            self.expert_data = pickle.load(f)
        self.num_samples = num_samples

    @property
    def requires_keys(self) -> Set[str]:
        return {"obs", "nxt"}
    
    @property
    def produces_keys(self) -> Set[str]:
        return {"exp_obs", "exp_nxt"}
    
    def __call__(self, data: MultiTensor) -> MultiTensor:
        # get expert obs pairs
        exp_idx = th.randperm(len(self.expert_data))
        data.exp_obs = self.expert_data.obs[exp_idx][:self.num_samples]
        data.exp_nxt = self.expert_data.nxt[exp_idx][:self.num_samples]
        return data