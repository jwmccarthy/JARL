import torch as th

from jarl.collect.behavior import Behavior
from jarl.collect.capture import CaptureContext, build_record, validate_captures
from jarl.data.records import EnvStep


class Runner:
    def __init__(self, env, behavior: Behavior, sink, captures=()) -> None:
        self.env = env
        self.behavior = behavior
        self.sink = sink
        self.captures = tuple(captures)
        validate_captures(self.captures)
        self.obs = None
        self.state = None

    @property
    def n_envs(self) -> int:
        return self.env.n_envs

    def reset(self):
        self.obs = self.env.reset()
        self.state = self.behavior.initial_state(self.n_envs)
        return self.obs

    @th.no_grad()
    def step(self) -> EnvStep:
        if self.obs is None:
            raise RuntimeError("runner must be reset before stepping")

        obs = th.as_tensor(self.obs, device=self.behavior.device)
        decision = self.behavior.act(obs, self.state)
        env_step = self.env.step(decision.action)
        context = CaptureContext(obs, self.state, decision, env_step)
        self.sink.append(build_record(context, self.captures))

        self.obs = env_step.collector_obs
        self.state = _reset_state(decision.next_state, env_step.done)
        return env_step


def _reset_state(state: th.Tensor | None, done) -> th.Tensor | None:
    if state is None:
        return None
    done = th.as_tensor(done, dtype=th.bool, device=state.device)
    if not done.any():
        return state
    state = state.clone()
    state[done] = 0
    return state
