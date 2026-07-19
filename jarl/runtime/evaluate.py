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
        steps: int = int(25e3),
    ) -> None:
        self.env = env
        self.policy = policy
        self.path = path
        self.freq = freq
        self.steps = steps
        self.current_step = 0

    def ready(self, step: int) -> bool:
        self.current_step = step
        return step % self.freq == 0

    def save(self) -> None:
        th.save(self.policy.state_dict(), f"{self.path}_{self.current_step}.pt")

    @th.no_grad()
    def run(self) -> None:
        obs = self.env.reset()
        for _ in range(self.steps):
            obs = th.as_tensor(obs, device=self.policy.device)
            decision = self.policy.act(obs, deterministic=True)
            obs = self.env.step(decision.action).collector_obs
        if self.path:
            self.save()
