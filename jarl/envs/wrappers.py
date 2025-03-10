import numpy as np
import torch as th
import torchvision.transforms.v2.functional as F

import gymnasium as gym
from typing import Tuple, Any
from gymnasium.spaces import Box

from jarl.data.dict import DotDict
from jarl.data.types import EnvStep
from jarl.envs.space import BoxSpace, ConcatSpace


class EpisodeStatsEnv(gym.Wrapper):

    def __init__(self, env: gym.Env) -> None:
        super().__init__(env)
        self.reward = self.length = 0

    def reset(self, seed: int = None, **kwargs) -> Any:
        self.reward = self.length = 0
        return super().reset(seed=seed, **kwargs)
    
    def step(self, act: np.ndarray):
        obs, rew, trm, trc, _ = super().step(act)
        self.reward += rew
        self.length += 1
        info = DotDict()
        if np.logical_or(trm, trc):
            info.update(rew=self.reward, len=self.length)
        return obs, rew, trm, trc, info
    

# class TorchEnvWrapper:

#     def __init__(self, env: TorchEnv) -> None:
#         self.env = env

#     def __getattr__(self, key: str) -> Any:
#         try:
#             return super().__getattr__(key)
#         except AttributeError:
#             return getattr(self.env, key)
    

# class FrameStackWrapper(TorchEnvWrapper):

#     def __init__(self, env: TorchEnv, n: int = 4) -> None:
#         super().__init__(env)
#         self.n = n
#         self.obs_space = ConcatSpace(self.obs_space, n, device=self.device)
#         self.frames = th.zeros(self.obs_space.shape, device=self.device)

#     def step(self, act: np.ndarray) -> EnvStep:
#         obs, rew, trc, don, nxt, info = self.env.step(act)
#         self.frames = th.roll(self.frames, shifts=-1, dims=0)
#         self.frames[-1] = obs
#         return self.frames.clone(), rew, trc, don, nxt, info
    
#     def reset(self) -> th.Tensor:
#         obs = self.env.reset()
#         self.frames.fill_(0)
#         self.frames[-1] = obs
#         return self.frames.clone()
    

# class ImageTransformWrapper(TorchEnvWrapper):

#     def __init__(
#         self, 
#         env: TorchEnv, 
#         size: Tuple[int, int] = (84, 84),
#         gray: bool = True
#     ) -> None:
#         super().__init__(env)  # Ensure recursive attribute access
#         self.size = size
#         self.gray = gray

#         # Refactor observation space
#         chans = 1 if gray else self.obs_space.shape[-1]
#         shape = self.obs_space.shape[:-3] + (chans, *self.size)
#         space = Box(low=0, high=1, shape=shape, dtype=np.float32)
#         self.obs_space = BoxSpace(space, device=self.device)

#     def _transform(self, obs: th.Tensor) -> th.Tensor:
#         # Ensure obs is in (C, H, W) format before transformations
#         obs = obs.movedim(-1, 0) / 255.0  # Convert to (C, H, W) format

#         # Perform transformations directly on GPU
#         if self.gray:
#             obs = F.rgb_to_grayscale(obs)  # Keeps on GPU
#         obs = F.resize(obs, self.size, antialias=True)  # Keeps on GPU
#         return obs

#     def reset(self) -> th.Tensor:
#         return self._transform(self.env.reset())

#     def step(self, act: np.ndarray) -> EnvStep:
#         obs, rew, trc, don, nxt, info = self.env.step(act)
#         return self._transform(obs), rew, trc, don, self._transform(nxt), info
