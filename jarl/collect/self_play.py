import copy
import random
import torch as th

from pathlib import Path
from collections import OrderedDict

from jarl.data.records import PolicyOutput
from jarl.collect.capture import CaptureContext, build_record
from jarl.collect.runner import _make_env_step, _reset_state


class SnapshotPool:

    def __init__(
        self,
        policy,
        max_size:          int,
        snapshot_interval: int,
        active_cache_size: int = 4,
        seed:              int = 0,
        checkpoint_dir:    Path | str | None = Path("./checkpoints/"),
    ) -> None:
        if max_size < 3:
            raise ValueError("snapshot pool must retain at least three policies")
        if snapshot_interval < 1 or active_cache_size < 1:
            raise ValueError("snapshot pool settings must be positive")

        self.max_size = max_size
        self.snapshot_interval = snapshot_interval
        self.active_cache_size = active_cache_size
        self._random = random.Random(seed)
        self.checkpoint_dir = (
            None if checkpoint_dir is None else Path(checkpoint_dir)
        )

        self._make_checkpoint_dir()

        self._archive = {}
        self._snapshots = OrderedDict()
        self._active = OrderedDict()
        self._next_id = 0
        self._last_snapshot = 0

        self.add(policy, timesteps=0)

    @property
    def ids(self) -> tuple[int, ...]:
        return tuple(self._snapshots)

    @property
    def archive_ids(self) -> tuple[int, ...]:
        return tuple(self._archive)

    def _make_checkpoint_dir(self) -> None:
        if self.checkpoint_dir is None:
            return
        if self.checkpoint_dir.exists() and any(self.checkpoint_dir.iterdir()):
            raise ValueError(
                f"snapshot directory is not empty: {self.checkpoint_dir}"
            )
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def add(
        self,
        policy,
        timesteps:     int,
        protected_ids: tuple[int, ...] = (),
    ) -> int:
        snapshot = copy.deepcopy(policy).eval().requires_grad_(False)
        if self.checkpoint_dir is None:
            snapshot.to("cpu")

        snapshot_id = self._next_id
        self._next_id += 1
        self._last_snapshot = timesteps
        self._snapshots[snapshot_id] = snapshot

        path = None
        if self.checkpoint_dir is not None:
            path = self.checkpoint_dir / f"policy_{timesteps:012d}.pt"
            temporary = path.with_suffix(".pt.tmp")
            th.save(snapshot.state_dict(), temporary)
            temporary.replace(path)
        self._archive[snapshot_id] = (timesteps, path)

        protected = set(protected_ids) | {snapshot_id}
        while len(self._snapshots) > self.max_size:
            old_id = next(
                (key for key in self._snapshots if key not in protected), None
            )
            if old_id is None:
                break
            self._snapshots.pop(old_id)
            self._active.pop(old_id, None)
            if self.checkpoint_dir is None:
                self._archive.pop(old_id)

        return snapshot_id

    def _opponent_candidates(self) -> list[int]:
        ids = list(self._snapshots)
        # Keep snapshot zero in the evaluation archive, but remove it from the
        # training pool once a trained policy exists.
        return ids[1:] if len(ids) > 1 and ids[0] == 0 else ids

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
        """Sample without replacement, weighting newer snapshots by rank."""
        if count < 1:
            raise ValueError("historical policy count must be positive")

        selected = []
        ids = self._opponent_candidates()

        for _ in range(min(count, len(ids))):
            index = self._random.choices(
                range(len(ids)), weights=range(1, len(ids) + 1), k=1
            )[0]
            selected.append(ids.pop(index))

        return tuple(selected)

    def select_ids(self, count: int) -> tuple[int, ...]:
        """Select the most recent policies for the active opponent window."""
        if count < 1:
            raise ValueError("historical policy count must be positive")
        return tuple(self._opponent_candidates()[-count:])

    def timesteps(self, snapshot_id: int) -> int:
        try:
            return self._archive[snapshot_id][0]
        except KeyError:
            raise KeyError(f"unknown snapshot {snapshot_id}") from None

    def policy(self, snapshot_id: int, device: th.device | str):
        archive = self._archive.get(snapshot_id)
        if archive is None:
            raise KeyError(f"unknown snapshot {snapshot_id}")

        cached = self._active.pop(snapshot_id, None)
        if cached is not None:
            self._active[snapshot_id] = cached
            return cached

        snapshot = self._snapshots.get(snapshot_id)
        if snapshot is not None:
            policy = copy.deepcopy(snapshot)
        else:
            policy = copy.deepcopy(next(iter(self._snapshots.values())))
            policy.load_state_dict(
                th.load(archive[1], map_location="cpu", weights_only=True)
            )

        policy = policy.to(device).eval()
        self._active[snapshot_id] = policy

        while len(self._active) > self.active_cache_size:
            self._active.popitem(last=False)

        return policy


class SelfPlayMatchmaker:
    """Assign either the learner or a historical policy to each team."""

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
        self._generator = th.Generator(device=self.device).manual_seed(seed)
        self.learner_mask = th.ones(self.n_envs, dtype=th.bool, device=self.device)
        self.opponent_ids = th.full(
            (self.n_envs,), -1, dtype=th.int64, device=self.device
        )
        self.learner_count = self.n_envs
        self.set_historical_ids(historical_ids)
        self.rematch()

    @property
    def historical_ids(self) -> tuple[int, ...]:
        return tuple(self._historical_ids.tolist())

    def set_historical_ids(self, historical_ids: tuple[int, ...]) -> None:
        if self.historical_fraction and not historical_ids:
            raise ValueError("historical self-play requires at least one snapshot")

        self._historical_ids = th.tensor(
            historical_ids, dtype=th.int64, device=self.device
        )
        self._historical_weights = th.arange(
            1, len(historical_ids) + 1, dtype=th.float32, device=self.device
        )

    def rematch(self, done: th.Tensor | None = None) -> None:
        if done is None:
            matches = th.arange(self.num_matches, device=self.device)
        else:
            matches = (
                th.as_tensor(done, dtype=th.bool, device=self.device)
                .view(self.num_matches, self.players_per_match)
                .any(-1)
                .nonzero(as_tuple=True)[0]
            )

        if not matches.numel():
            return

        shape = (self.num_matches, self.players_per_match)
        learner = self.learner_mask.view(shape)
        opponents = self.opponent_ids.view(shape)
        learner[matches] = True
        opponents[matches] = -1

        historical_matches = matches[
            th.rand(len(matches), generator=self._generator, device=self.device)
            < self.historical_fraction
        ]
        if not historical_matches.numel():
            self.learner_count = int(self.learner_mask.sum().item())
            return

        learner_team = th.randint(
            0, 2,
            (len(historical_matches),),
            generator=self._generator,
            device=self.device,
        )
        selected = th.multinomial(
            self._historical_weights,
            len(historical_matches),
            replacement=True,
            generator=self._generator,
        )
        snapshot_ids = self._historical_ids[selected]

        start = 0
        for team, size in enumerate(self.team_sizes):
            end = start + size
            rows = learner_team != team
            learner[historical_matches[rows], start:end] = False
            opponents[historical_matches[rows], start:end] = snapshot_ids[rows, None]
            start = end

        self.learner_count = int(self.learner_mask.sum().item())


class SelfPlayRunner:
    """Collect experience while routing historical actors to frozen policies."""

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

        for capture in self.captures:
            capture.reset(self.n_envs)

        self.matchmaker.rematch()
        self._timestep_count = self.matchmaker.learner_count
        return self.observation

    @th.no_grad()
    def step(self):
        if self.observation is None:
            raise RuntimeError("runner must be reset before stepping")

        observation = th.as_tensor(self.observation, device=self.policy.device)
        self._timestep_count = self.matchmaker.learner_count
        output = self._act(observation)
        env_step = _make_env_step(self.env.step(output.action))
        env_step.episode_groups = self._episode_groups()
        env_step.info = self._learner_episode_info(env_step)

        context = CaptureContext(observation, self.state, output, env_step)
        record = build_record(context, self.captures)
        record["learner_mask"] = self.matchmaker.learner_mask
        self.buffer.append(record)

        self.observation = env_step.observation
        self.state = _reset_state(output.next_state, env_step.done)
        self.matchmaker.rematch(env_step.done)
        return env_step

    def _episode_groups(self) -> dict[str, th.Tensor]:
        historical = self._historical_mask()
        learner = self.matchmaker.learner_mask

        return {
            "current":    learner & ~historical,
            "historical": learner & historical,
        }

    def _historical_mask(self) -> th.Tensor:
        historical_matches = (
            self.matchmaker.opponent_ids.view(
                self.matchmaker.num_matches, self.matchmaker.players_per_match
            )
            .ge(0)
            .any(-1)
        )

        return historical_matches.repeat_interleave(
            self.matchmaker.players_per_match
        )

    def _learner_episode_info(self, env_step) -> dict:
        finished = env_step.done.nonzero(as_tuple=True)[0]
        if not len(finished):
            return env_step.info

        learner = self.matchmaker.learner_mask[finished].cpu().tolist()
        historical = self._historical_mask()[finished].cpu().tolist()

        info = dict(env_step.info)
        for key, values in tuple(info.items()):
            if not isinstance(values, list) or len(values) != len(learner):
                continue

            info[key] = [value for value, keep in zip(values, learner) if keep]

            if key in ("reward", "length"):
                info[f"historical_{key}"] = [
                    value
                    for value, keep, historical in zip(values, learner, historical)
                    if keep and historical
                ]
                
        return info

    def after_update(self, timesteps: int) -> None:
        added = self.opponent_pool.maybe_add(
            self.snapshot_policy,
            timesteps,
            protected_ids=self.matchmaker.historical_ids,
        )

        if not added:
            return
    
        self.matchmaker.set_historical_ids(
            self.opponent_pool.select_ids(self.historical_policies)
        )

    def _state_for(self, mask: th.Tensor):
        return None if self.state is None else self.state[mask]

    def _route(
        self,
        value: th.Tensor,
        mask:  th.Tensor,
        fill:  int | None,
        device=None,
    ) -> th.Tensor:
        shape = (self.n_envs, *value.shape[1:])
        routed = (
            th.empty(shape, dtype=value.dtype, device=device or value.device)
            if fill is None
            else th.full(shape, fill, dtype=value.dtype, device=device or value.device)
        )
        routed[mask] = value

        return routed

    def _next_state(self, output: PolicyOutput, learner_mask: th.Tensor):
        if self.state is None:
            return None
        
        next_state = th.empty_like(self.state)
        next_state[learner_mask] = output.next_state

        return next_state

    def _act(self, observation: th.Tensor) -> PolicyOutput:
        learner_mask = self.matchmaker.learner_mask
        learner = self.policy.act(observation[learner_mask], self._state_for(learner_mask))
        if learner.log_prob is None:
            raise ValueError("learner policy did not produce log probabilities")

        action = self._route(learner.action, learner_mask, None, observation.device)
        log_prob = self._route(learner.log_prob, learner_mask, 0, observation.device)
        next_state = self._next_state(learner, learner_mask)
        extras = {
            key: self._route(value, learner_mask, 0)
            for key, value in learner.extras.items()
        }
        extras["learner_mask"] = learner_mask

        opponent_ids = self.matchmaker.opponent_ids
        for snapshot_id in opponent_ids[~learner_mask].unique().tolist():
            mask = opponent_ids == snapshot_id
            opponent = self.opponent_pool.policy(snapshot_id, observation.device)
            output = opponent.act(observation[mask], self._state_for(mask))
            action[mask] = output.action
            if next_state is not None:
                next_state[mask] = output.next_state

        return PolicyOutput(
            action=action,
            next_state=next_state,
            log_prob=log_prob,
            extras=extras,
        )
