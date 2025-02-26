import torch as th
from torch import Tensor

import gymnasium as gym

from jarl.data.dict import DotDict
from jarl.envs.gym import SyncVecEnv, AsyncVecEnv
from jarl.envs.space import StackedSpace, torch_space
from jarl.data.types import Device, GymStepOutput


class TorchGymEnv:
    """Gym env w/ torch tensor IO"""

    def __init__(
        self, 
        env_func: str,
        n_envs: int = 1,
        device: Device = "cpu",
        sync_env: bool = True,
        **kwargs: dict
    ) -> None:
        self.n_envs = n_envs
        self.device = device

        # initialize vectorized env
        env_funcs = n_envs * [env_func]
        env_type = SyncVecEnv if sync_env else AsyncVecEnv
        self.env = env_type(env_funcs)

        # torch observation and action spaces
        self.obs_space = torch_space(
            self.env.single_observation_space, device)
        self.act_space = torch_space(
            self.env.single_action_space, device)
        
        # vectorized spaces
        self.vec_obs_space = StackedSpace(
            self.obs_space, n_envs, device=self.obs_space.device)
        self.vec_act_space = StackedSpace(
            self.act_space, n_envs, device=self.act_space.device)

    def seed(self, seed: int) -> None:
        self.env.seed(seed)

    def reset(self) -> Tensor:
        obs, _ = self.env.reset()
        return self.vec_obs_space(obs)
    
    def _step(self, action: Tensor):
        if isinstance(action, Tensor):
            action = action.detach().cpu().numpy()
        return self.env.step(action)
    
    def step(self, trs: DotDict, stop: bool = False) -> GymStepOutput:     
        obs, rew, trm, trc, nxt = self._step(trs.act)

        # step output to tensors
        trs.rew = th.as_tensor(rew, dtype=th.float32)
        trs.trc = th.as_tensor(trc | stop)
        trs.don = th.as_tensor(trm | trc)
        trs.nxt = self.vec_obs_space(nxt)  # next observation (pre-reset)
        nxt = self.vec_obs_space(obs)      # next observation (post-reset)

        return trs, nxt
    
    def close(self) -> None:
        self.env.close()