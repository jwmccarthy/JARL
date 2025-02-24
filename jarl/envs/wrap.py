import numpy as np
import torch as th
from torchvision.transforms import v2

from gymnasium import spaces
from typing import Any, Tuple
from collections import deque

from jarl.data.dict import DotDict
from jarl.data.types import GymStepOutput

from jarl.envs.vec import TorchGymEnv
from jarl.envs.space import BoxSpace, ConcatSpace, StackedSpace


class TorchGymEnvWrapper:

    def __init__(self, env: TorchGymEnv) -> None:
        self.env = env

    def __getattr__(self, key: str) -> Any:
        if key in self.__dict__:
            return self[key]
        return getattr(self.env, key)
    
    def reset(self) -> th.Tensor:
        return self.env.reset()
    
    def step(self, trs: DotDict, stop: bool = False) -> GymStepOutput:
        return self.env.step(trs, stop)


class FormatImageWrapper(TorchGymEnvWrapper):

    def __init__(
        self, 
        env: TorchGymEnv | TorchGymEnvWrapper, 
        size: int | Tuple[int, int] = 84
    ) -> None:
        assert isinstance(env.obs_space, BoxSpace), (
            "Image formatting requires Box observation space"
        )

        super().__init__(env)
        self.size = size if isinstance(size, tuple) else (size, size)
        self.transform = v2.Resize(self.size)

        # new obs space image dims
        chans = self.obs_space.shape[-1]
        shape = self.obs_space.shape[:-3] + (chans, *self.size)
        space = spaces.Box(low=0, high=1, shape=shape, dtype=np.float32)
        self.obs_space = BoxSpace(space, device=self.device)
        self.vec_obs_space = StackedSpace(
            self.obs_space, self.n_envs, device=self.device)

    def _transform(self, obs: th.Tensor) -> th.Tensor:
        return self.transform(obs.movedim(-1, -3) / 255.0)
    
    def reset(self) -> th.Tensor:
        return self._transform(super().reset())
    
    def step(self, trs: DotDict, stop: bool = False) -> GymStepOutput:
        trs, obs = self.env.step(trs, stop)
        trs.nxt = self._transform(obs)
        obs = self._transform(obs)
        return trs, obs


class ObsStackWrapper(TorchGymEnvWrapper):

    def __init__(
        self, 
        env: TorchGymEnv | TorchGymEnvWrapper, 
        stack_len: int = 4
    ) -> None:
        super().__init__(env)
        self.stack_len = stack_len
        self.obs_queue = deque(maxlen=stack_len)
        self.obs_space = ConcatSpace(
            env.obs_space, stack_len, device=self.device)
        self.vec_obs_space = StackedSpace(
            self.obs_space, self.n_envs, device=self.device)

    def _stack(self) -> th.Tensor:
        return th.cat(list(self.obs_queue), dim=1)

    def _transform(self, obs: th.Tensor) -> th.Tensor:
        self.obs_queue.append(obs)
        return self._stack()

    def reset(self) -> th.Tensor:
        obs = super().reset()
        for _ in range(self.stack_len):
            self.obs_queue.append(obs)
        return self._stack()
    
    def step(self, trs: DotDict, stop: bool = False) -> GymStepOutput:
        trs, obs = super().step(trs, stop)
        trs.nxt = self._transform(trs.nxt)
        obs = self._transform(obs)
        return trs, obs
    

