import numpy as np
import torch as th

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
        curr_rews = self.env.n_envs * [0]
        curr_lens = self.env.n_envs * [0]

        for t in log:
            reset = False

            # step environment
            with th.no_grad():
                act = self.policy(obs)
            trs = DotDict(obs=obs, act=act)
            trs, obs = self.env.step(trs=trs)

            # store data
            self.buffer.store(trs)

            for i in range(self.env.n_envs):
                curr_rews[i] += trs.rew[i].item()
                curr_lens[i] += 1
                if trs.don[i]:
                    rews.append(curr_rews[i])
                    lens.append(curr_lens[i])
                    curr_rews[i] = curr_lens[i] = 0

            if (queue := self.ready(t)):
                log.update(episode=dict(
                    reward=np.mean(rews[-100:]),
                    length=np.mean(lens[-100:]),
                    glob_t=t*self.env.n_envs
                ))

            # run blocks
            for graph in queue:
                data = self.buffer.serve()
                info = graph.update(data)
                log.update(updates=info)

            if reset:
                obs = self.env.reset()

        return log