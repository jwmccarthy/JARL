import copy
import random
from collections import OrderedDict
from pathlib import Path

import torch as th

from jarl.collect.capture import CaptureContext, build_record
from jarl.collect.runner import _make_env_step, _reset_state
from jarl.data.records import PolicyOutput


class SnapshotPool:
    def __init__(
        self,
        policy,
        max_size:          int,
        snapshot_interval: int,
        active_cache_size: int = 4,
        seed: int = 0,
        checkpoint_dir: Path | None = None,
    ) -> None:
        if max_size < 1 or snapshot_interval < 1 or active_cache_size < 1:
            raise ValueError("snapshot pool settings must be positive")

        self.max_size = max_size
        self.snapshot_interval = snapshot_interval
        self.active_cache_size = active_cache_size
        self._random = random.Random(seed)
        self.checkpoint_dir = checkpoint_dir
        if self.checkpoint_dir is not None:
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self._snapshots = OrderedDict()
        self._active = OrderedDict()
        self._next_id = 0
        self._last_snapshot = 0
        self.add(policy, timesteps=0)

    @property
    def ids(self) -> tuple[int, ...]:
        return tuple(self._snapshots)

    def add(
        self,
        policy,
        timesteps:     int,
        protected_ids: tuple[int, ...] = (),
    ) -> int:
        snapshot = copy.deepcopy(policy).to("cpu").eval().requires_grad_(False)
        snapshot_id = self._next_id
        self._next_id += 1
        self._snapshots[snapshot_id] = snapshot
        self._last_snapshot = timesteps
        if self.checkpoint_dir is not None:
            th.save(
                snapshot.state_dict(),
                self.checkpoint_dir / f"policy_{timesteps:012d}.pt",
            )

        protected = set(protected_ids) | {snapshot_id}
        removable = [key for key in self._snapshots if key not in protected]
        while len(self._snapshots) > self.max_size and removable:
            removed_id = removable.pop(0)
            self._snapshots.pop(removed_id)
            self._active.pop(removed_id, None)

        return snapshot_id

    def maybe_add(
        self,
        policy,
        timesteps:     int,
        protected_ids: tuple[int, ...] = (),
    ) -> bool:
        if timesteps - self._last_snapshot < self.snapshot_interval:
            return False
        self.add(policy, timesteps, protected_ids)
        return True

    def sample_ids(self, count: int) -> tuple[int, ...]:
        if count < 1:
            raise ValueError("historical policy count must be positive")
        ids = list(self._snapshots)
        return tuple(self._random.sample(ids, min(count, len(ids))))

    def policy(self, snapshot_id: int, device: th.device | str):
        if snapshot_id not in self._snapshots:
            raise KeyError(f"unknown snapshot {snapshot_id}")
        if snapshot_id in self._active:
            policy = self._active.pop(snapshot_id)
            self._active[snapshot_id] = policy
            return policy

        policy = copy.deepcopy(self._snapshots[snapshot_id]).to(device).eval()
        self._active[snapshot_id] = policy
        while len(self._active) > self.active_cache_size:
            self._active.popitem(last=False)
        return policy


class SelfPlayMatchmaker:
    def __init__(
        self,
        num_matches:      int,
        team_sizes:       tuple[int, int],
        current_fraction: float,
        historical_ids:   tuple[int, ...],
        device:           th.device | str,
        seed:             int = 0,
    ) -> None:
        if num_matches < 1 or any(size < 1 for size in team_sizes):
            raise ValueError("match dimensions must be positive")
        if not 0.0 <= current_fraction <= 1.0:
            raise ValueError("current self-play fraction must be between zero and one")
        if current_fraction < 1.0 and not historical_ids:
            raise ValueError("historical self-play requires at least one snapshot")

        self.num_matches = num_matches
        self.team_sizes = team_sizes
        self.players_per_match = sum(team_sizes)
        self.n_envs = num_matches * self.players_per_match
        self.current_fraction = current_fraction
        self.historical_fraction = 1.0 - current_fraction
        self.device = th.device(device)
        self._historical_ids = th.as_tensor(
            historical_ids, dtype=th.int64, device=self.device
        )
        self._generator = th.Generator(device=self.device).manual_seed(seed)
        self.learner_mask = th.ones(self.n_envs, dtype=th.bool, device=self.device)
        self.opponent_ids = th.full(
            (self.n_envs,), -1, dtype=th.int64, device=self.device
        )
        self.learner_count = self.n_envs
        self.rematch()

    def set_historical_ids(self, historical_ids: tuple[int, ...]) -> None:
        if self.historical_fraction and not historical_ids:
            raise ValueError("historical self-play requires at least one snapshot")
        self._historical_ids = th.as_tensor(
            historical_ids, dtype=th.int64, device=self.device
        )

    @property
    def historical_ids(self) -> tuple[int, ...]:
        return tuple(self._historical_ids.tolist())

    def rematch(self, done: th.Tensor | None = None) -> None:
        if done is None:
            matches = th.arange(self.num_matches, device=self.device)
        else:
            done = th.as_tensor(done, dtype=th.bool, device=self.device)
            if done.shape != (self.n_envs,):
                raise ValueError(f"expected done shape {(self.n_envs,)}, got {done.shape}")
            match_done = done.view(self.num_matches, self.players_per_match).any(-1)
            matches = match_done.nonzero(as_tuple=True)[0]

        if not matches.numel():
            return

        learner = self.learner_mask.view(self.num_matches, self.players_per_match)
        opponents = self.opponent_ids.view(self.num_matches, self.players_per_match)
        learner[matches] = True
        opponents[matches] = -1

        historical = th.rand(
            len(matches), generator=self._generator, device=self.device
        ) < self.historical_fraction
        historical_matches = matches[historical]
        if not historical_matches.numel():
            self.learner_count = int(self.learner_mask.sum().item())
            return

        learner_team = th.randint(
            0,
            2,
            (len(historical_matches),),
            generator=self._generator,
            device=self.device,
        )
        selected = th.randint(
            0,
            len(self._historical_ids),
            (len(historical_matches),),
            generator=self._generator,
            device=self.device,
        )
        snapshot_ids = self._historical_ids[selected]

        left = 0
        for team, size in enumerate(self.team_sizes):
            right = left + size
            opponent_matches = historical_matches[learner_team != team]
            opponent_snapshots = snapshot_ids[learner_team != team]
            learner[opponent_matches, left:right] = False
            opponents[opponent_matches, left:right] = opponent_snapshots[:, None]
            left = right
        self.learner_count = int(self.learner_mask.sum().item())


class SelfPlayRunner:
    def __init__(
        self,
        env,
        policy,
        buffer,
        opponent_pool:       SnapshotPool,
        matchmaker:          SelfPlayMatchmaker,
        snapshot_policy,
        historical_policies: int = 1,
        captures=(),
    ) -> None:
        if historical_policies < 1:
            raise ValueError("historical_policies must be positive")
        if env.n_envs != matchmaker.n_envs:
            raise ValueError("environment and matchmaker actor counts differ")

        self.env = env
        self.policy = policy
        self.buffer = buffer
        self.opponent_pool = opponent_pool
        self.matchmaker = matchmaker
        self.snapshot_policy = snapshot_policy
        self.historical_policies = historical_policies
        self.captures = tuple(captures)
        self.observation = None
        self.state = None
        self._timestep_count = 0

    @property
    def n_envs(self) -> int:
        return self.env.n_envs

    @property
    def timestep_count(self) -> int:
        return self._timestep_count

    def reset(self):
        self.observation = self.env.reset()
        self.state = self.policy.initial_state(self.n_envs)
        self.matchmaker.rematch()
        self._timestep_count = self.matchmaker.learner_count
        return self.observation

    @th.no_grad()
    def step(self):
        if self.observation is None:
            raise RuntimeError("runner must be reset before stepping")

        observation = th.as_tensor(self.observation, device=self.policy.device)
        self._timestep_count = self.matchmaker.learner_count
        policy_output = self._act(observation)
        env_step = _make_env_step(self.env.step(policy_output.action))
        env_step.info = self._learner_episode_info(env_step)
        context = CaptureContext(observation, self.state, policy_output, env_step)
        record = build_record(context, self.captures)
        record["learner_mask"] = self.matchmaker.learner_mask
        self.buffer.append(record)

        self.observation = env_step.observation
        self.state = _reset_state(policy_output.next_state, env_step.done)
        self.matchmaker.rematch(env_step.done)
        return env_step

    def _learner_episode_info(self, env_step) -> dict:
        finished = env_step.done.nonzero(as_tuple=True)[0]
        if not len(finished):
            return env_step.info
        learner = self.matchmaker.learner_mask[finished].cpu().tolist()
        historical_matches = self.matchmaker.opponent_ids.view(
            self.matchmaker.num_matches,
            self.matchmaker.players_per_match,
        ).ge(0).any(dim=-1)
        historical = historical_matches[:, None].expand(
            -1, self.matchmaker.players_per_match
        ).reshape(-1)[finished].cpu().tolist()
        info = dict(env_step.info)
        for key in ("reward", "length"):
            values = info.get(key)
            if values is not None and len(values) == len(learner):
                info[key] = [
                    value for value, keep in zip(values, learner) if keep
                ]
                info[f"historical_{key}"] = [
                    value
                    for value, active, past in zip(values, learner, historical)
                    if active and past
                ]
        return info

    def after_update(self, timesteps: int) -> None:
        assigned = self.matchmaker.opponent_ids
        assigned = assigned[assigned >= 0].unique().tolist()
        protected_ids = tuple(set(assigned) | set(self.matchmaker.historical_ids))
        if self.opponent_pool.maybe_add(
            self.snapshot_policy, timesteps, protected_ids=protected_ids
        ):
            historical_ids = self.opponent_pool.sample_ids(self.historical_policies)
            self.matchmaker.set_historical_ids(historical_ids)

    def _act(self, observation: th.Tensor) -> PolicyOutput:
        learner_mask = self.matchmaker.learner_mask
        learner_state = None if self.state is None else self.state[learner_mask]
        learner_output = self.policy.act(observation[learner_mask], learner_state)
        if learner_output.log_prob is None:
            raise ValueError("learner policy did not produce log probabilities")

        action = th.empty(
            (self.n_envs, *learner_output.action.shape[1:]),
            dtype=learner_output.action.dtype,
            device=observation.device,
        )
        log_prob = th.zeros(
            (self.n_envs, *learner_output.log_prob.shape[1:]),
            dtype=learner_output.log_prob.dtype,
            device=observation.device,
        )
        action[learner_mask] = learner_output.action
        log_prob[learner_mask] = learner_output.log_prob
        next_state = self._next_state(learner_output, learner_mask)
        extras = {
            key: self._route_extra(value, learner_mask)
            for key, value in learner_output.extras.items()
        }
        extras["learner_mask"] = learner_mask

        opponent_ids = self.matchmaker.opponent_ids
        for snapshot_id in opponent_ids[~learner_mask].unique().tolist():
            opponent_mask = opponent_ids == snapshot_id
            opponent = self.opponent_pool.policy(snapshot_id, observation.device)
            opponent_state = None if self.state is None else self.state[opponent_mask]
            opponent_output = opponent.act(
                observation[opponent_mask], opponent_state
            )
            action[opponent_mask] = opponent_output.action
            if next_state is not None:
                if opponent_output.next_state is None:
                    raise ValueError("recurrent opponent did not produce a next state")
                next_state[opponent_mask] = opponent_output.next_state

        return PolicyOutput(
            action=action,
            next_state=next_state,
            log_prob=log_prob,
            extras=extras,
        )

    def _route_extra(
        self,
        value:        th.Tensor,
        learner_mask: th.Tensor,
    ) -> th.Tensor:
        routed = th.zeros(
            (self.n_envs, *value.shape[1:]),
            dtype=value.dtype,
            device=value.device,
        )
        routed[learner_mask] = value
        return routed

    def _next_state(
        self,
        learner_output: PolicyOutput,
        learner_mask: th.Tensor,
    ) -> th.Tensor | None:
        if self.state is None:
            if learner_output.next_state is not None:
                raise ValueError("feed-forward policy unexpectedly produced state")
            return None
        if learner_output.next_state is None:
            raise ValueError("recurrent learner did not produce a next state")

        next_state = th.empty_like(self.state)
        next_state[learner_mask] = learner_output.next_state
        return next_state
