from dataclasses import dataclass, field
from typing import Any

import torch as th


@dataclass
class PolicyOutput:
    action:     th.Tensor
    next_state: th.Tensor | None = None
    log_prob:   th.Tensor | None = None


@dataclass
class Evaluation:
    log_prob: th.Tensor | None = None
    entropy:  th.Tensor | None = None
    value:    th.Tensor | None = None
    q:        th.Tensor | tuple[th.Tensor, ...] | None = None
    extras:   dict[str, th.Tensor] = field(default_factory=dict)


@dataclass
class EnvStep:
    next_obs:    Any
    observation: Any
    reward:      Any
    terminated:  Any
    truncated:   Any
    info:        dict[str, Any] = field(default_factory=dict)

    @property
    def done(self):
        return self.terminated | self.truncated
