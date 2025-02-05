import torch as th

import pickle
from typing import Set

from jarl.data.multi import MultiTensor
from jarl.train.modify.base import DataModifier


class CatExpertObs(DataModifier):

    def __init__(
        self,
        data: MultiTensor = None, 
        path: str = None,
        num_samples: int = 2048
    ) -> None:
        if data is not None:
            self.expert_data = data
        if path is not None:
            with open(path, "rb") as f:
                self.expert_data = pickle.load(f)
        self.num_samples = num_samples

    @property
    def requires_keys(self) -> Set[str]:
        return {"obs", "next_obs"}
    
    @property
    def produces_keys(self) -> Set[str]:
        return {"pol_obs", "exp_obs"}
    
    def __call__(self, data: MultiTensor) -> MultiTensor:
        # get expert obs pairs
        exp_idx = th.randperm(len(self.expert_data))
        exp_obs = self.expert_data.obs[exp_idx][:self.num_samples]
        exp_next_obs = self.expert_data.next_obs[exp_idx][:self.num_samples]

        # concat with samples
        return MultiTensor(dict(
            pol_obs=th.cat([data.obs, data.next_obs], -1),
            exp_obs=th.cat([exp_obs, exp_next_obs], -1),
        ))