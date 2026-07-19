import numpy as np

import gymnasium as gym
from typing import Any

from jarl.data.dict import DotDict


class EpisodeStatsEnv(gym.Wrapper):

    def __init__(self, env: gym.Env) -> None:
        super().__init__(env)
        self.reward = self.length = 0

    def reset(self, seed: int = None, **kwargs) -> Any:
        self.reward = self.length = 0
        return super().reset(seed=seed, **kwargs)

    def step(self, action: np.ndarray):
        observation, reward, terminated, truncated, _ = super().step(action)
        self.reward += reward
        self.length += 1
        info = DotDict()
        if np.logical_or(terminated, truncated):
            info.update(reward=self.reward, length=self.length)
        return observation, reward, terminated, truncated, info


class ReshapeImageEnv(gym.Wrapper):

    def __init__(self, env: gym.Env) -> None:
        super().__init__(env)

        # roll stacked frames into channels
        total_shape = env.observation_space.shape
        image_shape = (total_shape[-1], *total_shape[-3:-1])
        stack_shape = total_shape[0] if len(total_shape) > 3 else 1

        # new observation space shape
        self.observation_space = gym.spaces.Box(
            low=0, high=255,
            shape=(image_shape[0] * stack_shape, *image_shape[1:]),
            dtype=np.uint8
        )

    def reset(self, seed: int = None, **kwargs) -> Any:
        observation, info = super().reset(seed=seed, **kwargs)
        return observation.reshape(self.observation_space.shape), info

    def step(self, action: np.ndarray):
        observation, *rest = super().step(action)
        return observation.reshape(self.observation_space.shape), *rest