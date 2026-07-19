from jarl.store.rollout import RolloutBuffer


class OnPolicySchedule:
    def ready(self, source: RolloutBuffer, clock) -> bool:
        return source.full

    def acquire(self, source: RolloutBuffer):
        return source.finish()

    def after_update(self, source: RolloutBuffer) -> None:
        source.clear()

    def pending(self, source: RolloutBuffer) -> bool:
        return source.position > 0


class OffPolicySchedule:
    def __init__(
        self,
        learning_starts_env_steps: int,
        update_every_vector_steps: int = 1,
    ) -> None:
        self.learning_starts_env_steps = learning_starts_env_steps
        self.update_every_vector_steps = update_every_vector_steps

    def ready(self, source, clock) -> bool:
        return (
            clock.env_steps >= self.learning_starts_env_steps
            and clock.vector_steps % self.update_every_vector_steps == 0
        )

    def acquire(self, source):
        return source

    def after_update(self, source) -> None:
        return None

    def pending(self, source) -> bool:
        return False
