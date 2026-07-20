from dataclasses import dataclass


@dataclass
class Clock:
    vector_steps:     int = 0
    env_steps:        int = 0
    episodes:         int = 0
    learner_updates: int = 0
    optimizer_steps: int = 0
