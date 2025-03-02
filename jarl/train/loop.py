import numpy as np
import torch as th

import time

from typing import List

from jarl.data.dict import DotDict
from jarl.data.buffer import Buffer

from jarl.envs.vec import TorchGymEnv
from jarl.modules.policy import Policy
from jarl.train.graph import TrainGraph
from jarl.log.log import Progress
from jarl.log.utils import episodic_return, episodic_length


class TrainLoop:

    def __init__(
        self, 
        env: TorchGymEnv, 
        buffer: Buffer,
        policy: Policy,
        graphs: List[TrainGraph],
        logger: Progress = Progress
    ) -> None:
        self.env = env
        self.buffer = buffer
        self.policy = policy
        self.logger = logger
        self.graphs = graphs

    def ready(self, t: int) -> List[TrainGraph]:
        return [g for g in self.graphs if g.ready(t)]

    def run(self, steps: int) -> None:
        obs = self.env.reset()
        log = self.logger(steps)
        rews, lens = [], []
        global_t = 0
        max_rew, max_len = -np.inf, -np.inf

        for g in self.graphs:
            g.init_schedulers(steps)

        start = time.time()

        for t in log:
            # step environment
            with th.no_grad():
                act = self.policy(obs)
            trs = DotDict(obs=obs, act=act)
            trs, obs, infos = self.env.step(trs=trs)

            global_t += self.env.n_envs

            # store data
            self.buffer.store(trs)

            # if "final_info" in infos:
            #     for info in infos["final_info"]:
            #         if info and "episode" in info:
            #             reward = info["episode"]["r"]
            #             length = info["episode"]["l"]
            #             rews.append(reward)
            #             lens.append(length)
            #             if reward > max_rew:
            #                 max_rew = reward
            #             if length > max_len:
            #                 max_len = length

            # log.update(episode=dict(
            #     reward=np.mean(rews[-50:]),
            #     length=np.mean(lens[-50:]),
            #     max_reward=max_rew,
            #     max_length=max_len,
            #     global_t=(t+1)*self.env.n_envs,
            #     time=time.time() - start
            # ))

            # run blocks
            for graph in self.ready(t):
                data = self.buffer.serve()
                info = graph.update(data)
                log.update(updates=info)

        return log