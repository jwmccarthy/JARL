import numpy as np
import torch as th
import torchvision.transforms.v2.functional as F

from typing import Tuple, Any
from gymnasium.spaces import Box

from jarl.data.types import EnvStep
from jarl.envs.gym import TorchEnv
from jarl.envs.space import BoxSpace, ConcatSpace


class TorchEnvWrapper:

    def __init__(self, env: TorchEnv) -> None:
        self.env = env

    def __getattr__(self, key: str) -> Any:
        try:
            return super().__getattr__(key)
        except AttributeError:
            return getattr(self.env, key)


class FireResetWrapper(TorchEnvWrapper):

    def __init__(self, env: TorchEnv) -> None:
        self.env = env

    def reset(self) -> th.Tensor:
        self.env.reset()
        obs, _, _, don, *_ = self.env.step(np.array(1))
        if don:
            self.env.reset()
        obs, _, _, don, *_ = self.env.step(np.array(2))
        if don:
            self.env.reset()
        return obs
    

class NoopResetWrapper(TorchEnvWrapper):

    def __init__(self, env: TorchEnv, noop_max: int = 30) -> None:
        self.env = env
        self.noop_max = noop_max

    def reset(self) -> th.Tensor:
        self.env.reset()
        noops = np.random.randint(1, self.noop_max + 1)
        for i in range(noops):
            obs, _, _, don, *_ = self.env.step(np.array(0))
            if don:
                obs = self.env.reset()
        return obs


class EpisodicLifeWrapper(TorchEnvWrapper):

    def __init__(self, env: TorchEnv) -> None:
        self.env = env
        self.lives = 0
        self.real_done = True

    def step(self, act: np.ndarray) -> th.Tensor:
        *exp, don, nxt, info = self.env.step(act)
        self.real_done = don
        lives = self.env.unwrapped.ale.lives()
        don = True if 0 < lives < self.lives else don
        self.lives = lives
        return *exp, don, nxt, info
    
    def reset(self) -> th.Tensor:
        if self.real_done:
            obs = self.env.reset()
        else:
            obs, _, _, don, *_ = self.env.step(np.array(0))
            obs = self.env.reset() if don else obs
        self.lives = self.env.unwrapped.ale.lives()
        return obs
    

class MaxAndSkipWrapper(TorchEnvWrapper):

    def __init__(self, env: TorchEnv, skip: int = 4) -> None:
        super().__init__(env)
        self._skip = skip
        self._frames = th.empty(
            (2, *self.obs_space.shape), device=self.device)
        self._frames_ptr = 0

    def step(self, act: np.ndarray) -> EnvStep:
        total_rew = 0

        for _ in range(self._skip):
            obs, rew, trc, don, nxt, info = self.env.step(act)
            total_rew += rew

            # rolling buffer
            self._frames[self._frames_ptr].copy_(obs)
            self._frames_ptr = 1 - self._frames_ptr

            if don: break

        # Compute max frame without tuple overhead
        max_frame = th.maximum(self._frames[0], self._frames[1])

        return max_frame, total_rew, trc, don, nxt, info

    

class FrameStackWrapper(TorchEnvWrapper):

    def __init__(self, env: TorchEnv, n: int = 4) -> None:
        super().__init__(env)
        self.n = n
        self.obs_space = ConcatSpace(self.obs_space, n, device=self.device)
        self.frames = th.zeros(self.obs_space.shape, device=self.device)

    def step(self, act: np.ndarray) -> EnvStep:
        obs, rew, trc, don, nxt, info = self.env.step(act)
        self.frames = th.roll(self.frames, shifts=-1, dims=0)
        self.frames[-1] = obs
        return self.frames.clone(), rew, trc, don, nxt, info
    
    def reset(self) -> th.Tensor:
        obs = self.env.reset()
        self.frames.fill_(0)
        self.frames[-1] = obs
        return self.frames.clone()
    

class ImageTransformWrapper(TorchEnvWrapper):

    def __init__(
        self, 
        env: TorchEnv, 
        size: Tuple[int, int] = (84, 84),
        gray: bool = True
    ) -> None:
        super().__init__(env)  # Ensure recursive attribute access
        self.size = size
        self.gray = gray

        # Refactor observation space
        chans = 1 if gray else self.obs_space.shape[-1]
        shape = self.obs_space.shape[:-3] + (chans, *self.size)
        space = Box(low=0, high=1, shape=shape, dtype=np.float32)
        self.obs_space = BoxSpace(space, device=self.device)

    def _transform(self, obs: th.Tensor) -> th.Tensor:
        # Ensure obs is in (C, H, W) format before transformations
        obs = obs.movedim(-1, 0) / 255.0  # Convert to (C, H, W) format

        # Perform transformations directly on GPU
        if self.gray:
            obs = F.rgb_to_grayscale(obs)  # Keeps on GPU
        obs = F.resize(obs, self.size, antialias=True)  # Keeps on GPU
        return obs

    def reset(self) -> th.Tensor:
        return self._transform(self.env.reset())

    def step(self, act: np.ndarray) -> EnvStep:
        obs, rew, trc, don, nxt, info = self.env.step(act)
        return self._transform(obs), rew, trc, don, self._transform(nxt), info
