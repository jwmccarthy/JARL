import math
from collections.abc import Callable
from dataclasses import dataclass


ValueSchedule = Callable[[float], float]
ValueSetter = Callable[[float], None]


@dataclass(frozen=True)
class ConstantSchedule:
    value: float

    def __call__(self, progress: float) -> float:
        return self.value


@dataclass(frozen=True)
class LinearSchedule:
    start: float
    end:   float

    def __call__(self, progress: float) -> float:
        return self.start + progress * (self.end - self.start)


@dataclass(frozen=True)
class MappedSchedule:
    schedule:  ValueSchedule
    transform: Callable[[float], float]

    def __call__(self, progress: float) -> float:
        return self.transform(self.schedule(progress))


@dataclass(frozen=True)
class ScheduledValue:
    name:     str
    schedule: ValueSchedule
    setter:   ValueSetter

    @classmethod
    def metric(
        cls,
        name:     str,
        schedule: ValueSchedule,
    ) -> "ScheduledValue":
        return cls(name=name, schedule=schedule, setter=lambda value: None)

    @classmethod
    def attribute(
        cls,
        name:      str,
        target,
        attribute: str,
        schedule:  ValueSchedule,
    ) -> "ScheduledValue":
        return cls(
            name=name,
            schedule=schedule,
            setter=lambda value: setattr(target, attribute, value),
        )

    def apply(self, progress: float) -> float:
        value = float(self.schedule(progress))
        if not math.isfinite(value):
            raise ValueError(f"scheduled value {self.name} is not finite")
        self.setter(value)
        return value


class ValueScheduler:
    def __init__(
        self,
        *values: ScheduledValue,
        section: str = "Schedule",
    ) -> None:
        if not values:
            raise ValueError("at least one scheduled value is required")
        names = [value.name for value in values]
        if len(names) != len(set(names)):
            raise ValueError("scheduled value names must be unique")

        self.values = values
        self.section = section
        self.total_steps = None
        self.current = {}

    def start(self, total_steps: int) -> None:
        if total_steps < 1:
            raise ValueError("schedule total steps must be positive")
        self.total_steps = total_steps
        self._apply(0.0)

    def advance(self, step: int) -> None:
        if self.total_steps is None:
            raise RuntimeError("value scheduler must be started before advancing")
        if step < 0:
            raise ValueError("schedule step cannot be negative")
        self._apply(min(step / self.total_steps, 1.0))

    def metrics(self) -> dict[str, dict[str, float]]:
        if self.total_steps is None:
            raise RuntimeError("value scheduler must be started before reporting")
        return {self.section: dict(self.current)}

    def _apply(self, progress: float) -> None:
        self.current = {
            "progress": progress,
            **{
                value.name: value.apply(progress)
                for value in self.values
            },
        }
