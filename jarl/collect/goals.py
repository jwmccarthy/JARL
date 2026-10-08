"""Goal-conditioned vector collection with replay and optional goal priors."""

from collections.abc import Callable, Mapping

import torch as th

from jarl.collect.runner import _make_env_step


class ReplayGoalSampler:
    """Sample achieved replay goals, optionally mixing in a CUDA goal dataset."""

    def __init__(
        self, replay, goal_from_observation: Callable[[th.Tensor], th.Tensor],
        *, expert_goals: th.Tensor | None = None, expert_fraction: float = 0.0,
        seed: int = 0,
    ) -> None:
        if not 0 <= expert_fraction <= 1:
            raise ValueError("expert-goal fraction must be in [0, 1]")
        if expert_fraction and expert_goals is None:
            raise ValueError("expert goals are required for a nonzero expert fraction")
        if expert_goals is not None and (
            expert_goals.ndim != 2 or not len(expert_goals)
            or expert_goals.shape[-1] != replay.achieved.shape[-1]
            or expert_goals.device != replay.device
        ):
            raise ValueError("expert goals must match the replay goal shape and device")
        self.replay = replay
        self.goal_from_observation = goal_from_observation
        self.expert_goals = expert_goals
        self.expert_fraction = expert_fraction
        self.generator = th.Generator(device=replay.device).manual_seed(seed)

    @th.no_grad()
    def __call__(self, count: int, observation: th.Tensor | None = None) -> th.Tensor:
        device = self.replay.device
        if self.replay.size:
            goals = self.replay.sample_goals(count)
        else:
            if observation is None or observation.device != device:
                raise ValueError("initial observations must be on the replay device")
            candidates = self.goal_from_observation(observation)
            if candidates.shape[-1] != self.replay.achieved.shape[-1]:
                raise ValueError("initial achieved goals do not match the replay")
            indices = th.randint(
                len(candidates), (count,), device=device, generator=self.generator,
            )
            goals = candidates[indices]

        if self.expert_fraction:
            indices = th.randint(
                len(self.expert_goals), (count,), device=device,
                generator=self.generator,
            )
            expert = self.expert_goals[indices]
            choose = th.rand(
                count, device=device, generator=self.generator,
            ) < self.expert_fraction
            goals = th.where(choose[:, None], expert, goals)
        return goals


class GoalConditionedRunner:
    """Collect GPU trajectories, command goals, and report goal diagnostics.

    The secondary JARL progress bar tracks replay prefill and collection blocks;
    an off-policy update can reuse it for minibatch progress between blocks.
    """

    def __init__(
        self, env, policy, buffer, goal_from_observation, goal_sampler,
        *, goal_horizon: int, goal_tolerance: float,
        components: Mapping[str, slice] | None = None,
        logger=None, report_interval: int = 8,
        prefill_steps: int = 1, collect_steps: int = 8,
        initial_vector_steps: int = 0,
    ) -> None:
        if min(goal_horizon, report_interval, prefill_steps, collect_steps) < 1:
            raise ValueError("goal and collection intervals must be positive")
        if goal_tolerance <= 0:
            raise ValueError("goal tolerance must be positive")
        self.env = env
        self.policy = policy
        self.buffer = buffer
        self.goal_from_observation = goal_from_observation
        self.goal_sampler = goal_sampler
        self.goal_horizon = goal_horizon
        self.goal_tolerance = goal_tolerance
        self.components = dict(components or {})
        self.logger = logger
        self.report_interval = report_interval
        self.prefill_steps = prefill_steps
        self.collect_steps = collect_steps
        self.vector_steps = initial_vector_steps
        self.observation: th.Tensor | None = None
        self.goals: th.Tensor | None = None
        self.goal_ages: th.Tensor | None = None
        self._metrics: dict[str, th.Tensor] = {}
        self.last_metrics: dict[str, float] = {}
        self._metric_steps = 0
        self._activity: str | None = None
        self._activity_total = 0
        self._activity_steps = 0

    @property
    def n_envs(self) -> int:
        return self.env.n_envs

    @property
    def timestep_count(self) -> int:
        return self.n_envs

    def reset(self) -> th.Tensor:
        observation = self.env.reset()
        if not isinstance(observation, th.Tensor) or observation.device != self.policy.device:
            raise ValueError("goal runner requires environment observations on the policy device")
        self.observation = observation
        self.goals = self.goal_sampler(self.n_envs, observation)
        if self.goals.device != observation.device:
            raise ValueError("sampled goals must be on the policy device")
        self.goal_ages = th.zeros(self.n_envs, dtype=th.long, device=observation.device)
        self._metrics.clear()
        self.last_metrics.clear()
        self._metric_steps = 0
        self._activity = None
        return observation

    def _start_collection(self) -> None:
        if self.logger is None or self._activity is not None:
            return
        if self.buffer.size < self.prefill_steps:
            total = self.prefill_steps - self.buffer.size
            activity = "prefill replay"
        else:
            total = self.collect_steps - self.vector_steps % self.collect_steps
            activity = "collect trajectories"
        self._activity = activity
        self._activity_total = total
        self._activity_steps = 0
        self.logger.start_activity(total, activity)

    def _finish_collection_step(self) -> None:
        if self._activity is None:
            return
        self._activity_steps += 1
        self.logger.advance_activity()
        if (self._activity_steps >= self._activity_total
                or self._activity == "prefill replay" and self.buffer.size >= self.prefill_steps):
            self.logger.finish_activity()
            self._activity = None

    @th.no_grad()
    def step(self):
        if self.observation is None or self.goals is None or self.goal_ages is None:
            raise RuntimeError("goal runner must be reset before stepping")
        self._start_collection()
        observation = self.observation
        action = self.policy.act(observation, self.goals)
        env_step = _make_env_step(self.env.step(action))
        if (not isinstance(env_step.next_obs, th.Tensor)
                or env_step.next_obs.device != observation.device
                or not isinstance(env_step.reward, th.Tensor)
                or env_step.reward.device != observation.device):
            raise ValueError("goal runner requires GPU-resident environment transitions")
        goal_error = self.goal_from_observation(env_step.next_obs) - self.goals
        distance = th.linalg.vector_norm(goal_error, dim=-1)
        reached = (
            th.ones_like(distance, dtype=th.bool) if self.components
            else distance < self.goal_tolerance
        )
        for name, coordinates in self.components.items():
            component_distance = th.linalg.vector_norm(goal_error[:, coordinates], dim=-1)
            component_reached = component_distance < self.goal_tolerance
            self._accumulate(f"{name}_goal_distance", component_distance.mean())
            self._accumulate(f"{name}_near_goal", component_reached.float().mean())
            reached &= component_reached

        self.goal_ages += 1
        boundary = env_step.done | (self.goal_ages >= self.goal_horizon)
        self.buffer.add(observation, action, env_step.next_obs, boundary)
        next_goals = self.goal_sampler(self.n_envs)
        if next_goals.device != observation.device:
            raise ValueError("sampled goals must remain on the policy device")
        self.goals = th.where(boundary[:, None], next_goals, self.goals)
        self.goal_ages.masked_fill_(boundary, 0)
        self.observation = env_step.observation
        self._accumulate("goal_distance", distance.mean())
        self._accumulate("near_goal", reached.float().mean())
        self._accumulate("reward_magnitude", th.as_tensor(env_step.reward).abs().float().mean())
        self._metric_steps += 1
        self.vector_steps += 1
        self._finish_collection_step()
        if self.logger is not None and self._metric_steps >= self.report_interval:
            self.logger.update(self.diagnostic_metrics(), step=self.vector_steps * self.n_envs)
        return env_step

    def _accumulate(self, key: str, value: th.Tensor) -> None:
        self._metrics[key] = self._metrics.get(key, 0) + value

    def diagnostic_metrics(self) -> dict[str, dict[str, float]]:
        if not self._metric_steps:
            return {}
        self.last_metrics = {
            key: (total / self._metric_steps).item()
            for key, total in self._metrics.items()
        }
        self.last_metrics["replay_steps"] = float(self.buffer.size)
        self._metrics.clear()
        self._metric_steps = 0
        return {"Goals": self.last_metrics}

    def after_update(self, env_steps: int) -> None:
        if self._activity is not None:
            self.logger.finish_activity()
            self._activity = None
