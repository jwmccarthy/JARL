from jarl.store.rollout import RolloutBuffer


class OnPolicySchedule:
    def expected_updates(
        self,
        vector_steps: int,
        environments_per_step: int,
        buffer: RolloutBuffer,
    ) -> int:
        return (vector_steps + buffer.horizon - 1) // buffer.horizon

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

    def expected_updates(
        self,
        vector_steps: int,
        environments_per_step: int,
        buffer,
    ) -> int:
        learning_starts = (
            self.learning_starts_env_steps + environments_per_step - 1
        ) // environments_per_step
        first_update = max(1, learning_starts)
        interval = self.update_every_vector_steps
        first_update = ((first_update + interval - 1) // interval) * interval

        if first_update > vector_steps:
            return 0

        return (vector_steps - first_update) // interval + 1

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
