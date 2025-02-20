import torch as th

from typing import Tuple
from collections import deque

from jarl.data.dict import DotDict
from jarl.envs.gym import TorchGymEnv
from jarl.envs.space import StackedSpace


class ObsStackWrapper:

    def __init__(
        self,
        env: TorchGymEnv,
        num_frame: int = 4
    ) -> None:
        self.env = env
        self.num_frame = num_frame
        self.obs_space = StackedSpace(env.obs_space, num_frame)
        self.obs_stack = th.zeros(self.obs_space.shape, device=self.device)

    def reset(self) -> th.Tensor:
        obs = self.env.reset()
        for i in range(self.num_frame):
            self.obs_stack[i] = obs
        return self._stack_obs()
    
    def step(
        self, 
        trs: DotDict = None, 
        stop: bool = False
    ) -> Tuple[DotDict[str, th.Tensor], th.Tensor]:     
        trs, obs = self.env.step(trs, stop)
        return