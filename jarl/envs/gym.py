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
        self.observation = np.empty(
            (n_envs, *self.obs_space.shape),
            dtype=np.float32,
        )
        self.reward = np.empty((n_envs,), dtype=np.float32)
        self.terminated = np.empty((n_envs,), dtype=bool)
        self.truncated = np.empty_like(self.terminated)
        self.next_observation = np.empty_like(self.observation)

    def reset(self) -> NDArray:
        observations = [env.reset()[0].astype(np.float32) for env in self.envs]
        return np.stack(observations)

    def step(self, action: NDArray | th.Tensor) -> EnvStep:
        actions = (
            action.detach().cpu().numpy()
            if isinstance(action, th.Tensor)
            else action
        )
        reward, length = [], []

        # step environments
        for index, (env, action) in enumerate(zip(self.envs, actions)):
            observation, current_reward, terminated, truncated, info = env.step(action)
            self.observation[index] = observation
            self.reward[index] = current_reward
            self.terminated[index] = terminated
            self.truncated[index] = truncated
            done = terminated | truncated
            self.next_observation[index] = (
                env.reset()[0] if done else observation
            )

            # pre-wrapper episodic reward
            if done and info:
                reward.append(info.reward)
                length.append(info.length)

        return EnvStep(
            next_obs=self.observation.copy(),
            observation=self.next_observation.copy(),
            reward=self.reward.copy(),
            terminated=self.terminated.copy(),
            truncated=self.truncated.copy(),
            info={"reward": reward, "length": length},
        )
