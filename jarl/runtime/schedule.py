from jarl.store.rollout import RolloutBuffer


class OnPolicySchedule:
    def ready(self, buffer: RolloutBuffer, clock) -> bool:
        return buffer.full

    def acquire(self, buffer: RolloutBuffer):
        return buffer.finish()

    def after_update(self, buffer: RolloutBuffer) -> None:
        buffer.clear()

    def pending(self, buffer: RolloutBuffer) -> bool:
        return buffer.position > 0


class OffPolicySchedule:
    def __init__(
        self,
        learning_starts_env_steps: int,
        update_every_vector_steps: int = 1,
    ) -> None:
        self.learning_starts_env_steps = learning_starts_env_steps
        self.update_every_vector_steps = update_every_vector_steps

    def ready(self, buffer, clock) -> bool:
        return (
            clock.env_steps >= self.learning_starts_env_steps
            and clock.vector_steps % self.update_every_vector_steps == 0
        )

    def acquire(self, buffer):
        return buffer

    def after_update(self, buffer) -> None:
        return None

    def pending(self, buffer) -> bool:
        return False
