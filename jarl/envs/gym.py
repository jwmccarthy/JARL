import numpy as np
from gymnasium.vector import SyncVectorEnv, AsyncVectorEnv


class SyncVecEnv(SyncVectorEnv):

    def __init__(self, env_fns):
        super().__init__(env_fns)

    def step(self, actions: np.ndarray) -> tuple:
        obs, rew, trm, trc, info = super().step(actions)
        nxt = np.zeros_like(obs)
        for i, d in enumerate(trm | trc):
            nxt[i] = info["final_observation"][i] if d else obs[i]
        return obs, rew, trm, trc, nxt, info
    

class AsyncVecEnv(AsyncVectorEnv):

    def __init__(self, env_fns):
        super().__init__(env_fns)

    def step(self, actions: np.ndarray) -> tuple:
        obs, rew, trm, trc, info = super().step(actions)
        nxt = np.zeros_like(obs)
        for i, d in enumerate(trm | trc):
            nxt[d] = info["final_observation"][i] if d else obs[i]
        return obs, rew, trm, trc, nxt