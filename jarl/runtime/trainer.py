from jarl.log.logger import Logger
from jarl.runtime.clock import Clock


class Trainer:
    def __init__(
        self,
        runner,
        buffer,
        learner,
        schedule,
        logger: Logger | None = None,
        checkpoint=None,
    ) -> None:
        self.runner = runner
        self.buffer = buffer
        self.learner = learner
        self.schedule = schedule
        self.logger = logger or Logger()
        self.checkpoint = checkpoint
        self.clock = Clock()

    def run(self, total_env_steps: int):
        vector_steps = total_env_steps // self.runner.n_envs
        if vector_steps < 1:
            raise ValueError("total_env_steps is smaller than one vector step")

        learner_updates = self.schedule.expected_updates(
            vector_steps,
            self.runner.n_envs,
            self.buffer,
        )
        self.runner.reset()

        for index in self.logger.progress(
            vector_steps,
            self.runner.n_envs,
            learner_updates,
        ):
            env_step = self.runner.step()
            self.clock.vector_steps += 1
            self.clock.env_steps += self.runner.n_envs
            self.clock.episodes += int(env_step.done.sum())
            self.logger.episode(self.clock.env_steps, env_step.info)

            if self.schedule.ready(self.buffer, self.clock):
                self._update()
            elif index == vector_steps - 1 and self.schedule.pending(self.buffer):
                self._update()

            if self.checkpoint and self.checkpoint.ready(self.clock.env_steps):
                self.checkpoint.run()

        return self.runner.policy

    def _update(self) -> None:
        data = self.schedule.acquire(self.buffer)
        metrics = self.learner.update(data)

        self.clock.learner_updates += 1
        self.logger.update(metrics, step=self.clock.env_steps)

        self.schedule.after_update(self.buffer)
