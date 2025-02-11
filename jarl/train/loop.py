import torch as th

from typing import Self, List

from jarl.data.dict import DotDict
from jarl.data.buffer import Buffer

from jarl.envs.gym import TorchGymEnv
from jarl.modules.policy import Policy
from jarl.train.graph import TrainGraph
from jarl.log.log import Progress


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

    def run(self, steps: int) -> None:
        obs = self.env.reset()
        log = self.logger(steps)

        for t in log:
            # step environment
            with th.no_grad():
                act = self.policy(obs)
            trs = DotDict(obs=obs, act=act)
            trs, obs = self.env.step(trs=trs)

            # store data
            self.buffer.store(trs)

            # run blocks
            for graph in [g for g in self.graphs if g.ready(t)]:
                data = self.buffer.serve()
                info = graph.update(data)
                log.update(updates=info)

        return log