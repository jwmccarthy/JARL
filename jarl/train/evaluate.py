import torch as th

from jarl.envs.gym import SyncGymEnv
from jarl.modules.policy import Policy


class Evaluator:

    def __init__(
        self, 
        env: SyncGymEnv, 
        policy: Policy,
        path: str = None,
        freq: int = int(25e4),
        steps: int = int(25e3)
    ) -> None:
        self.env = env
        self.policy = policy
        self.path = path
        self.freq = freq
        self.steps = steps
        self.current_step = 0

    def ready(self, t: int) -> bool:
        self.current_step = t
        return t % self.freq == 0

    def save(self) -> None:
        path = f"{self.path}_{self.current_step}.pt"
        th.save(self.policy.state_dict(), path)

    @th.no_grad()
    def run(self) -> None:
        rew = 0
        obs = self.env.reset()

        for _ in range(self.steps):
            obs = th.tensor(obs, device=self.policy.device)
            act = self.policy(obs, sample=False)
            _, obs, info = self.env.step(act)
            rew += sum(info["reward"])

        if self.path: self.save()