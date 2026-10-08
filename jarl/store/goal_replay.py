"""CUDA-native trajectory replay and future-goal relabeling for contrastive RL."""

from collections.abc import Callable
from dataclasses import dataclass

import torch as th

from jarl.store.replay import ReplayBuffer


@dataclass(frozen=True)
class ContrastiveBatch:
    observation: th.Tensor
    action: th.Tensor
    goal: th.Tensor
    steps_to_goal: th.Tensor


class FutureGoalReplayBuffer(ReplayBuffer):
    """Time-major replay with episode-safe discounted achieved-state goals.

    A slot contains one full vector step. The achieved goal comes from the
    actual post-action observation, including final_obs on a same-step reset.
    ``boundary`` can also cut trajectories when the commanded goal changes.
    Indices, sampling probabilities, stored transitions and minibatches all
    remain on the storage device.
    """

    def __init__(
        self,
        capacity: int,
        num_envs: int,
        observation_size: int,
        goal_size: int,
        action_shape: tuple[int, ...],
        goal_from_observation: Callable[[th.Tensor], th.Tensor],
        gamma: float,
        future_horizon: int,
        device: str | th.device,
        *,
        action_dtype: th.dtype = th.uint8,
    ) -> None:
        super().__init__(capacity, num_envs, device, device)
        if observation_size < 1 or goal_size < 1 or not action_shape:
            raise ValueError("observations, goals, and actions must have positive dimensions")
        if not 1 <= future_horizon <= capacity:
            raise ValueError("future horizon must fit replay capacity")
        if not 0 < gamma <= 1:
            raise ValueError("future-goal discount must be in (0, 1]")
        self.goal_from_observation = goal_from_observation
        self._storage = {
            "observation": th.empty(
                capacity, num_envs, observation_size, device=self.device,
            ),
            "action": th.empty(
                capacity, num_envs, *action_shape, dtype=action_dtype, device=self.device,
            ),
            "achieved_goal": th.empty(capacity, num_envs, goal_size, device=self.device),
            "episode": th.empty(capacity, num_envs, dtype=th.long, device=self.device),
        }
        self.current_episode = th.zeros(num_envs, dtype=th.long, device=self.device)
        self.future_weights = gamma ** th.arange(
            future_horizon, dtype=th.float32, device=self.device,
        )
        self.inserted = 0

    @property
    def observations(self) -> th.Tensor:
        return self._storage["observation"]

    @property
    def actions(self) -> th.Tensor:
        return self._storage["action"]

    @property
    def achieved(self) -> th.Tensor:
        return self._storage["achieved_goal"]

    @property
    def episodes(self) -> th.Tensor:
        return self._storage["episode"]

    @th.no_grad()
    def add(
        self, observation: th.Tensor, action: th.Tensor,
        transition_next: th.Tensor, boundary: th.Tensor,
    ) -> None:
        if (observation.shape != self.observations.shape[1:]
                or transition_next.shape != observation.shape
                or action.shape != self.actions.shape[1:]
                or boundary.shape != (self.num_envs,)):
            raise ValueError("replay transition must contain a complete vector step")
        if any(
            not isinstance(value, th.Tensor) or value.device != self.device
            for value in (observation, action, transition_next, boundary)
        ):
            raise ValueError("replay transitions must already be on the storage device")
        if boundary.dtype != th.bool:
            raise ValueError("trajectory boundaries must be bool tensors")
        achieved = self.goal_from_observation(transition_next)
        if (achieved.shape != self.achieved.shape[1:]
                or achieved.device != self.device):
            raise ValueError("achieved goals must have the configured shape and device")
        super().append({
            "observation": observation,
            "action": action.to(self.actions.dtype),
            "achieved_goal": achieved,
            "episode": self.current_episode,
        })
        self.current_episode.add_(boundary.long())
        self.inserted += 1

    def append(self, transition: dict[str, th.Tensor]) -> None:
        self.add(
            transition["observation"], transition["action"],
            transition["next_obs"], transition["boundary"],
        )

    @th.no_grad()
    def sample_goals(self, count: int) -> th.Tensor:
        if not self.size or count < 1:
            raise ValueError("cannot sample goals from an empty replay or an empty batch")
        times = th.randint(
            self.inserted - self.size, self.inserted, (count,), device=self.device,
        )
        environments = th.randint(self.num_envs, (count,), device=self.device)
        return self.achieved[times % self.capacity, environments]

    @th.no_grad()
    def sample(self, batch_size: int) -> ContrastiveBatch:
        if not self.size or batch_size < 1:
            raise ValueError("cannot sample an empty replay or an empty batch")
        times = th.randint(
            self.inserted - self.size, self.inserted, (batch_size,), device=self.device,
        )
        environments = th.randint(self.num_envs, (batch_size,), device=self.device)
        slots = times % self.capacity
        horizon = min(len(self.future_weights), self.size)
        offsets = th.arange(horizon, device=self.device)
        candidate_times = times[:, None] + offsets[None, :]
        future_slots = candidate_times % self.capacity
        same_episode = (
            (candidate_times < self.inserted)
            & (self.episodes[future_slots, environments[:, None]]
               == self.episodes[slots, environments][:, None])
        )
        # Offset zero is the state reached immediately after this action.
        weights = self.future_weights[:horizon] * same_episode
        offsets = th.multinomial(weights, 1).squeeze(-1)
        action = self.actions[slots, environments]
        if action.dtype == th.uint8:
            action = action.long()
        return ContrastiveBatch(
            self.observations[slots, environments], action,
            self.achieved[(times + offsets) % self.capacity, environments],
            offsets + 1,
        )
