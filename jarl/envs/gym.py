import torch as th
from torch import Tensor

from gymnasium import Env

from jarl.data.dict import DotDict
from jarl.envs.space import torch_space
from jarl.data.types import Device, GymStepOutput


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

    def reset(self) -> Tensor:
        obs, _ = self.env.reset()
        return self.obs_space(obs)
    
    def _step(self, action: Tensor):
        if isinstance(action, Tensor):
            action = action.detach().cpu().numpy()
        return self.env.step(action)[:-1]
    
    def step(self, trs: DotDict, stop: bool = False) -> GymStepOutput:     
        nxt, rew, trm, trc = self._step(trs.act)

        # step output to tensors
        trs.rew = th.as_tensor(rew, dtype=th.float32)
        trs.trc = th.as_tensor(trc | stop)
        trs.don = th.as_tensor(trm | trs.trc)
        trs.nxt = self.obs_space(nxt)  # next observation (pre-reset)

        # automatically reset
        nxt = self.reset() if trs.don else trs.nxt

        return trs, nxt
    
    def close(self) -> None:
        self.env.close()