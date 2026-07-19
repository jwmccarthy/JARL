import numpy as np
import torch as th

import gymnasium as gym
from typing import Callable, Any
from numpy.typing import NDArray

from jarl.data.records import EnvStep
from jarl.envs.space import torch_space


class SyncGymEnv:

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
        self.terminated = np.empty((n_envs,), dtype=bool)
        self.truncated = np.empty_like(self.terminated)
        self.collector_obs = np.empty_like(self.obs)

    def reset(self) -> NDArray:
        obs = [env.reset()[0].astype(np.float32) for env in self.envs]
        return np.stack(obs)

    def step(self, act: NDArray | th.Tensor) -> EnvStep:
        actions = act.detach().cpu().numpy() if isinstance(act, th.Tensor) else act
        reward, length = [], []

        # step environments
        for i, (env, act) in enumerate(zip(self.envs, actions)):
            obs, rew, trm, trc, info = env.step(act)
            self.obs[i] = obs
            self.rew[i] = rew
            self.terminated[i] = trm
            self.truncated[i] = trc
            done = trm | trc
            self.collector_obs[i] = env.reset()[0] if done else obs

            # pre-wrapper episodic reward
            if done and info:
                reward.append(info.rew)
                length.append(info.len)

        return EnvStep(
            next_obs=self.obs.copy(),
            collector_obs=self.collector_obs.copy(),
            reward=self.rew.copy(),
            terminated=self.terminated.copy(),
            truncated=self.truncated.copy(),
            info={"reward": reward, "length": length},
        )
