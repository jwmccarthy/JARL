import numpy as np
import torch as th

import gymnasium as gym
from typing import Callable, Any

from jarl.data.dict import DotDict
from jarl.data.types import Device, EnvOutput
from jarl.envs.space import torch_space


class SyncEnv:

    def __init__(
        self, 
        env_func: Callable[[Any], gym.Env], 
        n_envs: int = 1, 
        device: Device = "cpu"
    ) -> None:
        self.n_envs = n_envs
        self.device = device
        self.envs = [env_func() for _ in range(n_envs)]
        self.obs_space = torch_space(
            self.envs[0].observation_space, device)
        self.act_space = torch_space(
            self.envs[0].action_space, device)

        # storage for transition values
        self.obs = np.empty((n_envs, *self.obs_space.shape),
                            dtype=np.float32)
        self.rew = np.empty((n_envs,), dtype=np.float32)
        self.trc = np.empty((n_envs,), dtype=np.bool)
        self.don = np.empty_like(self.trc)
        self.nxt = np.empty_like(self.obs)

    def reset(self) -> th.Tensor:
        obs = np.stack([env.reset()[0] for env in self.envs])
        return th.as_tensor(obs).to(self.device)
    
    def step(self, trs: DotDict[str, th.Tensor]) -> EnvOutput:
        actions = trs.act.detach().cpu().numpy()

        # step environments
        for i, (env, act) in enumerate(zip(self.envs, actions)):
            obs, rew, trm, trc, _ = env.step(act)
            self.obs[i] = obs
            self.rew[i] = rew
            self.trc[i] = trc
            self.don[i] = trm | trc
            self.nxt[i] = env.reset()[0] if self.don[i] else obs

        # convert to tensor and store in transition
        trs.rew = th.tensor(self.rew, device=self.device)
        trs.trc = th.tensor(self.trc, device=self.device)
        trs.don = th.tensor(self.don, device=self.device)
        trs.nxt = th.tensor(self.obs, device=self.device)

        return trs, th.tensor(self.nxt, device=self.device)
