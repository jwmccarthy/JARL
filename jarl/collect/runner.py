import torch as th

from jarl.collect.capture import CaptureContext, build_record
from jarl.data.records import EnvStep
from jarl.modules.policy import Policy


class Runner:
    def __init__(self, env, policy: Policy, buffer, captures=()) -> None:
        self.env = env
        self.policy = policy
        self.buffer = buffer
        self.captures = tuple(captures)
        self.observation = None
        self.state = None

    @property
    def n_envs(self) -> int:
        return self.env.n_envs

    def reset(self):
        self.observation = self.env.reset()
        self.state = self.policy.initial_state(self.n_envs)
        return self.observation

    @th.no_grad()
    def step(self) -> EnvStep:
        if self.observation is None:
            raise RuntimeError("runner must be reset before stepping")

        observation = th.as_tensor(
            self.observation,
            device=self.policy.device,
        )
        policy_output = self.policy.act(observation, self.state)
        env_step = self.env.step(policy_output.action)
        context = CaptureContext(observation, self.state, policy_output, env_step)
        self.buffer.append(build_record(context, self.captures))

        self.observation = env_step.observation
        self.state = _reset_state(policy_output.next_state, env_step.done)
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
