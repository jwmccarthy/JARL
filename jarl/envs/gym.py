import torch as th
from torch import Tensor

from gymnasium import Env

from typing import Tuple

from jarl.data.dict import DotDict
from jarl.data.types import Device
from jarl.envs.space import torch_space


class TorchGymEnv:
    """Gym env w/ torch tensor IO"""

    def __init__(
        self, 
        env: Env,
        device: Device = "cpu"
    ) -> None:
        self.env = env
        self.device = device
        self.obs_space = torch_space(env.observation_space, device)
        self.act_space = torch_space(env.action_space, device)

    def seed(self, seed: int) -> None:
        self.env.seed(seed)

    # TODO: do I actually need info from reset?
    def reset(self) -> Tensor:
        obs, _ = self.env.reset()
        return self.obs_space(obs)
    
    def _step(self, action: Tensor):
        if isinstance(action, Tensor):
            action = action.detach().cpu().numpy()
        return self.env.step(action)[:-1]
    
    def step(
        self, 
        act: Tensor = None, 
        trs: DotDict = None, 
        stop: bool = False
    ) -> Tuple[DotDict[str, Tensor], Tensor]:     
        assert (act is None) != (trs is None)

        # init transition if needed
        trs = trs or DotDict(act=act)

        obs, rew, trm, trc = self._step(trs.act)

        # non-space vals to tensors
        trs.rew = th.as_tensor(rew, dtype=th.float32)
        trs.trc = th.as_tensor(trc | stop)
        trs.don = th.as_tensor(trm | trs.trc)

        # next observation to tensor
        trs.nxt = self.obs_space(obs)

        # automatically reset
        obs = self.reset() if trs.don else trs.nxt

        return trs, obs