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
        value_scheduler=None,
        update_callback=None,
    ) -> None:
        self.runner = runner
        self.buffer = buffer
        self.learner = learner
        self.schedule = schedule
        self.logger = logger or Logger()
        self.checkpoint = checkpoint
        self.value_scheduler = value_scheduler
        self.update_callback = update_callback
        self.clock = Clock()

    def run(self, total_timesteps: int):
        if total_timesteps < 1:
            raise ValueError("total_timesteps must be positive")
        if self.value_scheduler is not None:
            self.value_scheduler.start(total_timesteps)
            self.value_scheduler.advance(self.clock.env_steps)
        self.runner.reset()
        if total_timesteps < self.runner.timestep_count:
            raise ValueError("total_timesteps is smaller than one vector step")

        with self.logger.progress(
            total_timesteps,
            initial_timesteps=self.clock.env_steps,
        ):
            while self.clock.env_steps < total_timesteps:
                self._step(total_timesteps)

        return self.runner.policy

    def _step(self, total_timesteps: int) -> None:
        env_step = self.runner.step()
        timesteps = self.runner.timestep_count
        self.clock.vector_steps += 1
        self.clock.env_steps += timesteps
        self.clock.episodes += int(env_step.done.sum())
        self.logger.advance(timesteps)
        self.logger.episode(self.clock.env_steps, env_step.info)
        if self.value_scheduler is not None:
            self.value_scheduler.advance(self.clock.env_steps)

        if self.schedule.ready(self.buffer, self.clock):
            self._update()
        elif (
            self.clock.env_steps >= total_timesteps
            and self.schedule.pending(self.buffer)
        ):
            self._update()

        if self.checkpoint and self.checkpoint.ready(self.clock.env_steps):
            self.checkpoint.run()

    def _update(self) -> None:
        data = self.schedule.acquire(self.buffer)
        metrics = self.learner.update(data)
        if self.value_scheduler is not None:
            scheduled_metrics = self.value_scheduler.metrics()
            overlap = metrics.keys() & scheduled_metrics.keys()
            if overlap:
                raise ValueError(
                    f"scheduled metric section already exists: {', '.join(overlap)}"
                )
            metrics.update(scheduled_metrics)

        self.clock.learner_updates += 1
        self.logger.update(metrics, step=self.clock.env_steps)

        self.schedule.after_update(self.buffer)
        after_update = getattr(self.runner, "after_update", None)
        if after_update is not None:
            after_update(self.clock.env_steps)
        if self.update_callback is not None:
            self.update_callback(self)
