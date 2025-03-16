import pickle
import torch as th

from typing import Set

from jarl.data.multi import MultiTensor
from jarl.modules.discrim import Discriminator
from jarl.train.modify.base import DataModifier


class CatExpertObs(DataModifier):

    def __init__(
        self, 
        demo_path: str,
        n_samples: int = None
    ) -> None:
        with open(demo_path, "rb") as f:
            self.exp_data = pickle.load(f)[self.requires_keys]
        self.n_samples = n_samples

    @property
    def requires_keys(self) -> Set[str]:
        return {"obs", "nxt"}

    @property
    def produces_keys(self) -> Set[str]:
        return {"obs", "nxt", "lbl"}

    def __call__(self, data: MultiTensor) -> MultiTensor:
        exp_data = (
            self.exp_data if self.n_samples
            else self.exp_data.sample(self.n_samples)
        )
        pol_targ = th.ones(len(data))
        exp_targ = th.zeros(len(exp_data))
        all_data = data[self.requires_keys].append(exp_data)
        all_data.lbl = th.cat([pol_targ, exp_targ])
        return all_data
    

class ComputeExpertObsReward(DataModifier):

    def __init__(self, discrim: Discriminator) -> None:
        self.discrim = discrim

    @property
    def requires_keys(self) -> Set[str]:
        return {"obs", "rew", "nxt"}

    @property
    def produces_keys(self) -> Set[str]:
        return {"rew"}

    def __call__(self, data: MultiTensor) -> MultiTensor:
        data.rew = -th.log(self.discrim((data.obs, data.nxt)))
        return data