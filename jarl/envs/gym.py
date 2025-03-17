import numpy as np
import torch as th

import gymnasium as gym
from typing import Callable, Any
from numpy.typing import NDArray

from jarl.data.dict import DotDict
from jarl.data.types import EnvOutput
from jarl.envs.space import torch_space


class SyncEnv:

    def __init__(
        self, 
        env_func: Callable[[Any], gym.Env], 
        n_envs: int = 1
    ) -> None:
        self.n_envs = n_envs
        self.envs = [env_func() for _ in range(n_envs)]

        # wrap env spaces
        self.obs_space = torch_space(self.envs[0].observation_space)
        self.act_space = torch_space(self.envs[0].action_space)

        # storage for transition values
        self.obs = np.empty((n_envs, *self.obs_space.shape),
                            dtype=np.float32)
        self.rew = np.empty((n_envs,), dtype=np.float32)
        self.trc = np.empty((n_envs,), dtype=np.bool)
        self.don = np.empty_like(self.trc)
        self.nxt = np.empty_like(self.obs)

    def reset(self) -> th.Tensor:
        obs = [env.reset()[0].astype(np.float32) for env in self.envs]
        return np.stack(obs)
    
    def step(self, act: NDArray | th.Tensor) -> EnvOutput:
        actions = act
        if isinstance(act, th.Tensor):
            actions = act.detach().cpu().numpy()
        reward, length = [], []   

        # step environments
        for i, (env, act) in enumerate(zip(self.envs, actions)):
            obs, rew, trm, trc, info = env.step(act)
            self.obs[i] = obs
            self.rew[i] = rew
            self.trc[i] = trc
            self.don[i] = trm | trc
            self.nxt[i] = env.reset()[0] if self.don[i] else obs

            # pre-wrapper episodic reward
            if self.don[i] and info:
                reward.append(info.rew)
                length.append(info.len)
                
        trs = dict(
            act=actions,
            rew=self.rew,
            trc=self.trc,
            don=self.don,
            nxt=self.obs
        )

        # episode statistics
        info = DotDict(reward=reward, length=length)

        return trs, np.copy(self.nxt), info
    
    def sample(self) -> NDArray:
        space = self.act_space.space
        return np.array([space.sample() for _ in range(self.n_envs)])