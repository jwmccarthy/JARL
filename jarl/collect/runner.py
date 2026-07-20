import numpy as np
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

    @property
    def timestep_count(self) -> int:
        return self.n_envs

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
        env_step = _make_env_step(self.env.step(policy_output.action))

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


def _make_env_step(result) -> EnvStep:
    observation, reward, terminated, truncated, info = result
    next_obs, final_available, info = _transition_observation(
        observation, terminated, truncated, info
    )

    if isinstance(observation, th.Tensor):
        terminated = th.as_tensor(terminated, dtype=th.bool, device=observation.device)
        truncated = th.as_tensor(truncated, dtype=th.bool, device=observation.device)
    else:
        terminated = np.asarray(terminated, dtype=bool)
        truncated = np.asarray(truncated, dtype=bool)

    bootstrap = ~terminated & (~truncated | final_available)
    return EnvStep(
        next_obs=next_obs,
        observation=observation,
        reward=reward,
        terminated=terminated,
        truncated=truncated,
        info=info,
        bootstrap=bootstrap,
    )


def _transition_observation(observation, terminated, truncated, info):
    info = dict(info)
    has_final = "final_obs" in info or "final_observation" in info
    if isinstance(observation, th.Tensor):
        next_obs = observation.clone() if has_final else observation
        done = th.as_tensor(
            terminated, dtype=th.bool, device=observation.device
        ) | th.as_tensor(truncated, dtype=th.bool, device=observation.device)
        available = th.zeros_like(done)
    else:
        next_obs = np.array(observation, copy=True) if has_final else observation
        done = np.asarray(terminated, dtype=bool) | np.asarray(truncated, dtype=bool)
        available = np.zeros_like(done)

    for final_key, mask_key in (
        ("final_obs", "_final_obs"),
        ("final_observation", "_final_observation"),
    ):
        if final_key not in info:
            continue

        final_obs = info.pop(final_key)
        mask = info.pop(mask_key, done)
        if isinstance(observation, th.Tensor):
            mask = th.as_tensor(mask, dtype=th.bool, device=observation.device)
            final_obs = th.as_tensor(final_obs, device=observation.device)
            next_obs[mask] = final_obs[mask]
        else:
            mask = np.asarray(mask, dtype=bool)
            final_obs = np.asarray(final_obs)
            if final_obs.dtype == object:
                for index in np.flatnonzero(mask):
                    next_obs[index] = final_obs[index]
            else:
                next_obs[mask] = final_obs[mask]
        available |= mask
        break

    return next_obs, available, info
