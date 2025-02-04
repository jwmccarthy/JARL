import torch as th

import pickle
from typing import Set

from jarl.data.multi import MultiTensor
from jarl.train.modify.base import DataModifier


class CatExpertObs(DataModifier):

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
        return {"obs", "next_obs"}
    
    @property
    def produces_keys(self) -> Set[str]:
        return {"obs", "next_obs", "lbl"}
    
    def __call__(self, data: MultiTensor) -> MultiTensor:
        # sample expert data
        exp_idx = th.randperm(len(self.expert_data))[:self.num_samples]
        exp_obs = self.expert_data.obs[exp_idx]
        exp_next_obs = self.expert_data.next_obs[exp_idx]
        exp_lbl = th.zeros(self.num_samples)

        # current policy samples
        pol_obs = data.obs
        pol_next_obs = data.next_obs
        pol_lbl = th.ones(len(data))

        # concat with samples
        return MultiTensor(dict(
            obs=th.cat([exp_obs, pol_obs]),
            next_obs=th.cat([exp_next_obs, pol_next_obs]),
            lbl=th.cat([exp_lbl, pol_lbl])
        ))