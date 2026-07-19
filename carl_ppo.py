#!/usr/bin/env python3
"""Train PPO with JARL on the CUDA-accelerated CARL environment."""

import argparse
from collections import deque
from datetime import datetime
import json
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical
from torch.optim import Adam
import trueskill

import carl

from jarl.collect import (
    CaptureContext,
    LogProbCapture,
    PolicyVersionCapture,
    RecurrentStateCapture,
    ValueCapture,
    build_record,
)
from jarl.data.records import EnvStep, Evaluation, PolicyOutput
from jarl.envs.space import torch_space
from jarl.learn import (
    OptimizerStep,
    PPOConfig,
    PPOLearner,
    PPOOptimizer,
    unique_parameters,
)
from jarl.log.logger import Logger
from jarl.modules.utils import init_layer
from jarl.sample import RecurrentRolloutMinibatches
from jarl.store import RolloutBuffer
from jarl.transform import GAE


ACTION_SIZES = (3, 3, 3, 2, 2, 3, 2)


class CarlEnv:
    """Small GPU-native adapter implementing the environment API JARL needs."""

    def __init__(
        self,
        n_sim: int,
        n_blue: int,
        n_orange: int,
        seed: int,
        max_ticks: int,
        reward_scale: float,
        tick_skip: int,
        team_spirit: float = 1.0,
    ) -> None:
        self.env = carl.Env(n_sim, n_blue, n_orange, seed, tick_skip)
        self.env.max_ticks = max_ticks
        self.n_sim = n_sim
        self.n_blue = n_blue
        self.n_orange = n_orange
        self.n_cars = self.env.n_cars
        self.n_agents = n_sim * self.n_cars
        self.n_envs = n_sim * n_blue
        self.max_ticks = max_ticks
        self.reward_scale = reward_scale
        self.tick_skip = tick_skip
        self.team_spirit = team_spirit
        self.previous_boost = torch.zeros((n_sim, self.n_cars), device="cuda")
        self.previous_demoed = torch.zeros(
            (n_sim, self.n_cars), dtype=torch.bool, device="cuda"
        )
        self.touch_decay = torch.ones((n_sim, self.n_cars), device="cuda")
        self.last_toucher = torch.full((n_sim,), -1, dtype=torch.long, device="cuda")
        self.previous_ball_distance = torch.zeros((n_sim, self.n_cars), device="cuda")
        self.previous_actions = torch.zeros(
            (n_sim, self.n_cars, len(ACTION_SIZES)),
            dtype=torch.long,
            device="cuda",
        )
        self.has_previous_action = torch.zeros(
            (n_sim, self.n_cars), dtype=torch.bool, device="cuda"
        )
        self.team_sign = torch.cat(
            (
                torch.ones(n_blue, device="cuda"),
                -torch.ones(n_orange, device="cuda"),
            )
        )

        blue = list(range(n_blue))
        orange = list(range(n_blue, self.n_cars))
        view_order = []
        for car in range(self.n_cars):
            teammates = blue if car < n_blue else orange
            opponents = orange if car < n_blue else blue
            view_order.append(
                [car] + [other for other in teammates if other != car] + opponents
            )
        self.view_order = torch.tensor(view_order, device="cuda")

        # 29 global values, 8 relative values for every other player, and
        # 26 values per player. CARL does not expose boost-pad or demo timers.
        self.obs_dim = 21 + 34 * self.n_cars
        observation = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.obs_dim,), dtype=np.float32
        )
        act = gym.spaces.MultiDiscrete(
            np.asarray(self.env.action_nvec[0], dtype=np.int32), dtype=np.int32
        )
        self.obs_space = torch_space(observation, device="cuda")
        self.act_space = torch_space(act, device="cuda")

    def raw_state(self, observation=None) -> torch.Tensor:
        if observation is None:
            observation = self.env.get_obs()
        raw = (
            observation.clone()
            if isinstance(observation, torch.Tensor)
            else torch.from_dlpack(observation).clone()
        )
        if raw.ndim == 3:
            # Blue observer zero orders cars as the global blue-then-orange state.
            raw = raw[:, 0]
        return raw

    def _build_observation(self, observation=None) -> torch.Tensor:
        raw = self.raw_state(observation)
        ball = raw[:, None, :9].expand(-1, self.n_cars, -1).clone()
        cars = raw[:, 9:].view(self.n_sim, self.n_cars, 21)
        cars = cars[:, self.view_order].clone()

        # Orange sees the same canonical attacking direction as blue: rotate
        # every world-space vector by pi around Z.
        ball[:, self.n_blue :, 0:2].neg_()
        ball[:, self.n_blue :, 3:5].neg_()
        ball[:, self.n_blue :, 6:8].neg_()
        orange_cars = cars[:, self.n_blue :]
        orange_cars[..., 0:2].neg_()
        orange_cars[..., 3:5].neg_()
        orange_cars[..., 6:8].neg_()
        orange_cars[..., 9:11].neg_()
        orange_cars[..., 12:14].neg_()

        ball_position = ball[..., 0:3]
        ball_velocity = ball[..., 3:6]
        car_position = cars[..., 0:3]
        car_velocity = cars[..., 3:6]
        car_angular_velocity = cars[..., 6:9]
        forward = cars[..., 9:12]
        up = cars[..., 12:15]

        yaw = torch.atan2(forward[..., 1], forward[..., 0])
        pitch = torch.asin(forward[..., 2].clamp(-1.0, 1.0))
        sin_roll = up[..., 1] * yaw.cos() - up[..., 0] * yaw.sin()
        cos_roll = up[..., 2] / pitch.cos().clamp_min(1e-6)
        roll = torch.atan2(sin_roll, cos_roll)
        rotation = torch.stack((pitch, yaw, roll), dim=-1) / torch.pi

        ball_delta = ball_position[:, :, None, :] - car_position
        velocity_to_ball = ball_velocity[:, :, None, :] - car_velocity
        speed = car_velocity.norm(dim=-1, keepdim=True)
        on_ground = cars[..., 16:17].bool()
        demoed = cars[..., 17:18].bool()
        has_flip = on_ground | ~(cars[..., 18:19].bool() | cars[..., 19:20].bool())
        player_features = torch.cat(
            (
                car_position / 6000.0,
                car_velocity / 2300.0,
                speed / 2300.0,
                car_angular_velocity / 6.0,
                rotation,
                (speed >= 2200.0).float(),
                ball_delta / 6000.0,
                ball_delta.norm(dim=-1, keepdim=True) / 6000.0,
                velocity_to_ball / 4600.0,
                velocity_to_ball.norm(dim=-1, keepdim=True) / 4600.0,
                (~demoed).float(),
                cars[..., 15:16] / 100.0,
                on_ground.float(),
                has_flip.float(),
            ),
            dim=-1,
        )

        relative_position = car_position[..., 1:, :] - car_position[..., :1, :]
        relative_velocity = car_velocity[..., 1:, :] - car_velocity[..., :1, :]
        relative_players = torch.cat(
            (
                relative_position / 6000.0,
                relative_position.norm(dim=-1, keepdim=True) / 6000.0,
                relative_velocity / 4600.0,
                relative_velocity.norm(dim=-1, keepdim=True) / 4600.0,
            ),
            dim=-1,
        ).flatten(2)

        previous_action = torch.cat(
            [
                torch.nn.functional.one_hot(self.previous_actions[..., index], size)
                for index, size in enumerate(ACTION_SIZES)
            ],
            dim=-1,
        ).float()
        previous_action *= self.has_previous_action[..., None]
        previous_action = torch.cat(
            (previous_action, self.has_previous_action[..., None].float()), dim=-1
        )

        game_features = torch.cat(
            (
                ball_position / 6000.0,
                ball_velocity / 6000.0,
                ball_velocity.norm(dim=-1, keepdim=True) / 6000.0,
                ball[..., 6:9] / 6.0,
                previous_action,
                relative_players,
            ),
            dim=-1,
        )

        observation = torch.cat((game_features, player_features.flatten(2)), dim=-1)
        return observation.flatten(0, 1)

    @staticmethod
    def _cosine(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return (
            torch.nn.functional.normalize(a, dim=-1)
            * torch.nn.functional.normalize(b, dim=-1)
        ).sum(dim=-1)

    def _reward(
        self,
        raw: torch.Tensor,
        score_delta: torch.Tensor,
        touch: torch.Tensor,
        touch_height: torch.Tensor,
        episode_done: torch.Tensor,
    ) -> torch.Tensor:
        ball_pos = raw[:, None, :3]
        ball_vel = raw[:, None, 3:6]
        cars = raw[:, 9:].view(self.n_sim, self.n_cars, 21)
        car_pos = cars[..., :3]
        car_vel = cars[..., 3:6]
        car_forward = cars[..., 9:12]
        boost = cars[..., 15].clamp(0.0, 100.0)
        demoed = cars[..., 17].bool()
        car_to_ball = ball_pos - car_pos
        distance = car_to_ball.norm(dim=-1)

        scoring_team = torch.where(
            score_delta[:, None] > 0,
            self.team_sign[None, :] > 0,
            self.team_sign[None, :] < 0,
        ) & (score_delta[:, None] != 0)
        scorer = torch.nn.functional.one_hot(
            self.last_toucher.clamp_min(0), self.n_cars
        ).bool()
        valid_scorer = (scorer & scoring_team).any(dim=1, keepdim=True)
        scorer = torch.where(valid_scorer, scorer & scoring_team, scoring_team)
        goal_scored = scorer.float() * (
            1.0 + 0.5 * ball_vel.norm(dim=-1).clamp_max(6000.0) / 6000.0
        )
        boost_difference = boost.sqrt() / 10.0 - self.previous_boost.sqrt() / 10.0
        boost_difference *= (~episode_done)[:, None]
        ball_touch = (
            touch.float()
            * self.touch_decay
            * ((touch_height + 91.25).clamp_min(0.0) / 182.5).pow(0.2836)
        )

        new_demo = demoed & ~self.previous_demoed
        demo = torch.cat(
            (
                new_demo[:, self.n_blue :]
                .any(dim=1, keepdim=True)
                .expand(-1, self.n_blue),
                new_demo[:, : self.n_blue]
                .any(dim=1, keepdim=True)
                .expand(-1, self.n_orange),
            ),
            dim=1,
        ).float()
        distance_progress = ((self.previous_ball_distance - distance) / 2300.0).clamp(
            -1.0, 1.0
        )
        distance_progress *= (~episode_done)[:, None]
        opponent_goal = torch.zeros_like(car_pos)
        opponent_goal[..., 1] = self.team_sign[None, :] * 5124.25
        own_goal = opponent_goal.clone()
        own_goal[..., 1].neg_()
        distance_ball_goal = torch.exp(
            -0.5 * ((ball_pos - opponent_goal).norm(dim=-1) - 966.0) / 6000.0
        ).clamp(0.0, 1.0)
        facing_ball = self._cosine(car_forward, car_to_ball)
        align_ball_goal = 0.5 * self._cosine(
            car_to_ball, car_pos - own_goal
        ) + 0.5 * self._cosine(-car_to_ball, opponent_goal - car_pos)
        touched_last = torch.nn.functional.one_hot(
            self.last_toucher.clamp_min(0), self.n_cars
        ).float()
        touched_last *= (self.last_toucher >= 0)[:, None]
        behind_ball = (
            (ball_pos[..., 1] - car_pos[..., 1]) * self.team_sign[None, :] >= 0
        ).float()
        velocity_player_ball = self._cosine(car_vel, car_to_ball)
        velocity = (car_vel.norm(dim=-1) / 2300.0).clamp(0.0, 1.0)
        boost_amount = boost.sqrt() / 10.0
        forward_velocity = ((car_forward * car_vel).sum(dim=-1) / 2300.0).clamp(
            -1.0, 1.0
        )

        reward = (
            1.45 * goal_scored
            + 0.1 * boost_difference
            + 0.1 * ball_touch
            + 0.3 * demo
            + 0.1 * distance_progress
            + 0.0025 * distance_ball_goal
            + 0.000625 * facing_ball
            + 0.0025 * align_ball_goal
            + 0.00125 * touched_last
            + 0.00125 * behind_ball
            + 0.00125 * velocity_player_ball
            + 0.000625 * velocity
            + 0.00125 * boost_amount
            + 0.0015 * forward_velocity
        )

        blue = reward[:, : self.n_blue]
        orange = reward[:, self.n_blue :]
        competitive = torch.cat(
            (
                blue - orange.mean(dim=1, keepdim=True),
                orange - blue.mean(dim=1, keepdim=True),
            ),
            dim=1,
        ).flatten()

        self.previous_boost.copy_(boost)
        self.previous_demoed.copy_(demoed)
        self.previous_ball_distance.copy_(distance)

        mixed = (1.0 - self.team_spirit) * reward.flatten()
        mixed += self.team_spirit * competitive
        return mixed * self.reward_scale

    def reset(self) -> torch.Tensor:
        self.env.reset()
        self.touch_decay.fill_(1.0)
        self.last_toucher.fill_(-1)
        self.previous_actions.zero_()
        self.has_previous_action.zero_()
        raw = self.raw_state()
        cars = raw[:, 9:].view(self.n_sim, self.n_cars, 21)
        self.previous_boost.copy_(cars[..., 15])
        self.previous_demoed.copy_(cars[..., 17].bool())
        self.previous_ball_distance.copy_(
            (cars[..., :3] - raw[:, None, :3]).norm(dim=-1)
        )
        return self._build_observation(raw)

    def step(self, actions: torch.Tensor):
        actions = actions.to(dtype=torch.int32).contiguous()
        carl_actions = actions.view(self.n_sim, self.n_cars, -1)
        raw_obs = self.raw_state(self.env.step(carl_actions))
        touch = torch.from_dlpack(self.env.get_ball_touches()).bool().clone()
        touch_height = raw_obs[:, None, 2].expand(-1, self.n_cars)
        toucher = torch.where(
            touch,
            torch.arange(self.n_cars, device="cuda")[None, :],
            -1,
        ).amax(dim=1)
        self.last_toucher.copy_(torch.where(toucher >= 0, toucher, self.last_toucher))
        score_delta = torch.from_dlpack(self.env.get_rewards()).clone()
        env_done = torch.from_dlpack(self.env.get_dones()).bool().clone()

        self.previous_actions.copy_(actions.view(self.n_sim, self.n_cars, -1))
        self.has_previous_action.fill_(True)
        self.previous_actions[env_done] = 0
        self.has_previous_action[env_done] = False

        next_obs = self._build_observation(raw_obs)
        goal = (score_delta[:, None] * self.team_sign).flatten()

        self.touch_decay.copy_(
            torch.where(
                touch,
                (self.touch_decay * 0.95).clamp_min(0.1),
                (self.touch_decay + 0.013).clamp_max(1.0),
            )
        )

        reward = self._reward(raw_obs, score_delta, touch, touch_height, env_done)
        self.touch_decay[env_done] = 1.0
        self.last_toucher[env_done] = -1

        done = env_done[:, None].expand(-1, self.n_cars).flatten()
        return EnvStep(
            next_obs=next_obs,
            observation=next_obs,
            reward=reward,
            terminated=done,
            truncated=torch.zeros_like(done),
            info={"goal": goal},
        )

    def team_data(self, value: torch.Tensor, blue: bool) -> torch.Tensor:
        value = value.view(self.n_sim, self.n_cars, *value.shape[1:])
        team = value[:, : self.n_blue] if blue else value[:, self.n_blue :]
        return team.flatten(0, 1)

    def combine_actions(
        self, learner: torch.Tensor, opponent: torch.Tensor
    ) -> torch.Tensor:
        learner = learner.view(self.n_sim, self.n_blue, -1)
        opponent = opponent.view(self.n_sim, self.n_orange, -1)
        return torch.cat((learner, opponent), dim=1).flatten(0, 1)


class GRUBackbone(nn.Module):
    def __init__(
        self,
        obs_dim: int,
        hidden_size: int,
        encoder: nn.Module = None,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.encoder = encoder or nn.Sequential(
            init_layer(nn.Linear(obs_dim, hidden_size)),
            nn.ReLU(),
        )
        self.gru = nn.GRU(hidden_size, hidden_size)
        for name, parameter in self.gru.named_parameters():
            if "weight" in name:
                nn.init.orthogonal_(parameter)
            else:
                nn.init.zeros_(parameter)

    def initial_state(self, batch_size: int, device="cuda") -> torch.Tensor:
        return torch.zeros(1, batch_size, self.hidden_size, device=device)

    def forward(
        self,
        observation: torch.Tensor,
        hidden: torch.Tensor,
        reset: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        encoded = self.encoder(observation)
        if reset is None or not reset.any():
            return self.gru(encoded, hidden)

        outputs = []
        for index in range(len(encoded)):
            hidden = hidden * (~reset[index])[None, :, None]
            output, hidden = self.gru(encoded[index : index + 1], hidden)
            outputs.append(output)
        return torch.cat(outputs), hidden


class GRUPolicy(nn.Module):
    def __init__(
        self,
        env: CarlEnv,
        hidden_size: int,
        backbone: GRUBackbone = None,
    ) -> None:
        super().__init__()
        self.version = 0
        self.backbone = backbone or GRUBackbone(env.obs_dim, hidden_size)
        nvec = env.act_space.nvec.flatten().tolist()
        self.sizes = tuple(int(value) for value in nvec)
        self.action_shape = env.act_space.shape
        self.self_player_offset = 29 + 8 * (env.n_cars - 1)
        self.output = nn.Sequential(
            init_layer(nn.Linear(hidden_size, hidden_size)),
            nn.LeakyReLU(),
            init_layer(nn.Linear(hidden_size, hidden_size // 2)),
            nn.LeakyReLU(),
            init_layer(nn.Linear(hidden_size // 2, sum(self.sizes)), std=0.01),
        )
        self.register_buffer(
            "inference_tick_skip", torch.tensor(env.tick_skip, dtype=torch.int32)
        )

    def increment_version(self) -> None:
        self.version += 1

    def initial_state(self, batch_size: int) -> torch.Tensor:
        return self.backbone.initial_state(
            batch_size, next(self.parameters()).device
        ).squeeze(0)

    @staticmethod
    def _disable(
        logits: torch.Tensor, action: int, unavailable: torch.Tensor
    ) -> torch.Tensor:
        masked = logits.clone()
        masked[..., action] = masked[..., action].masked_fill(unavailable, -1e8)
        return masked

    def _mask_actions(
        self, logits: tuple[torch.Tensor, ...], observation: torch.Tensor
    ) -> tuple[torch.Tensor, ...]:
        boost = observation[..., self.self_player_offset + 23]
        on_ground = observation[..., self.self_player_offset + 24] > 0.5
        has_flip = observation[..., self.self_player_offset + 25] > 0.5
        masked = list(logits)
        for action in (1, 2):
            masked[1] = self._disable(masked[1], action, on_ground)
            masked[5] = self._disable(masked[5], action, on_ground)
        masked[2] = self._disable(masked[2], 1, ~on_ground)
        masked[3] = self._disable(masked[3], 1, ~on_ground)
        masked[4] = self._disable(masked[4], 1, boost <= 1e-6)
        masked[6] = self._disable(masked[6], 1, ~has_flip)
        return tuple(masked)

    def _dist(
        self, observation: torch.Tensor, hidden: torch.Tensor
    ) -> tuple[list[Categorical], torch.Tensor]:
        features, next_hidden = self.backbone(observation, hidden)
        logits = self.output(features).split(self.sizes, dim=-1)
        logits = self._mask_actions(logits, observation)
        return [Categorical(logits=value) for value in logits], next_hidden

    def act(
        self,
        observation: torch.Tensor,
        state: torch.Tensor | None = None,
        *,
        deterministic: bool = False,
    ) -> PolicyOutput:
        if state is None:
            state = self.initial_state(len(observation))
        distributions, next_hidden = self._dist(observation.unsqueeze(0), state.unsqueeze(0))
        if deterministic:
            action = torch.stack(
                [dist.logits.argmax(dim=-1) for dist in distributions], -1
            )
        else:
            action = torch.stack([dist.sample() for dist in distributions], -1)
        action = action.reshape(*action.shape[:-1], *self.action_shape)
        logprob = torch.stack(
            [dist.log_prob(action[..., i]) for i, dist in enumerate(distributions)],
            dim=-1,
        ).sum(-1)
        return PolicyOutput(
            action=action.squeeze(0),
            next_state=next_hidden.squeeze(0),
            log_prob=logprob.squeeze(0),
        )

    def evaluate_actions(
        self,
        observation: torch.Tensor,
        action: torch.Tensor,
        state: torch.Tensor | None = None,
        *,
        reset: torch.Tensor | None = None,
    ) -> Evaluation:
        if state is None:
            state = self.initial_state(observation.shape[1])
        features, _ = self.backbone(observation, state.unsqueeze(0), reset)
        logits = self.output(features).split(self.sizes, dim=-1)
        logits = self._mask_actions(logits, observation)
        distributions = [Categorical(logits=value) for value in logits]
        flat_act = action.reshape(*action.shape[: -len(self.action_shape)], -1)
        logprob = torch.stack(
            [dist.log_prob(flat_act[..., i]) for i, dist in enumerate(distributions)],
            dim=-1,
        ).sum(-1)
        entropy = torch.stack([dist.entropy() for dist in distributions], dim=-1).sum(
            -1
        )
        return Evaluation(log_prob=logprob, entropy=entropy)


class GRUValue(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        backbone: GRUBackbone,
    ) -> None:
        super().__init__()
        self.version = 0
        self.backbone = backbone
        self.output = nn.Sequential(
            init_layer(nn.Linear(hidden_size, hidden_size // 2)),
            nn.LeakyReLU(),
            init_layer(nn.Linear(hidden_size // 2, hidden_size // 4)),
            nn.LeakyReLU(),
            init_layer(nn.Linear(hidden_size // 4, 1), std=1.0),
        )

    def increment_version(self) -> None:
        self.version += 1

    def value(
        self, observation: torch.Tensor, state: torch.Tensor | None = None
    ) -> torch.Tensor:
        if state is None:
            state = self.backbone.initial_state(
                len(observation), next(self.parameters()).device
            ).squeeze(0)
        features, _ = self.backbone(observation.unsqueeze(0), state.unsqueeze(0))
        return self.output(features).squeeze(0).squeeze(-1)

    def evaluate_values(
        self,
        observation: torch.Tensor,
        state: torch.Tensor | None = None,
        *,
        reset: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if state is None:
            state = self.backbone.initial_state(
                observation.shape[1], next(self.parameters()).device
            ).squeeze(0)
        features, _ = self.backbone(observation, state.unsqueeze(0), reset)
        return self.output(features).squeeze(-1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--total-timesteps", type=int, default=1_000_000_000)
    parser.add_argument("--n-sim", type=int, default=1024)
    parser.add_argument("--n-blue", type=int, default=1)
    parser.add_argument("--n-orange", type=int, default=1)
    parser.add_argument("--max-ticks", type=int, default=4096)
    parser.add_argument("--tick-skip", type=int, default=8)
    parser.add_argument("--rollout-steps", type=int, default=512)
    parser.add_argument("--sequence-len", type=int, default=32)
    parser.add_argument("--hidden-size", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=16384)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=2.5e-4)
    parser.add_argument("--ent-coef", type=float, default=1e-3)
    parser.add_argument("--reward-scale", type=float, default=1.0)
    parser.add_argument("--team-spirit", type=float, default=1.0)
    parser.add_argument("--gamma", type=float, default=0.999)
    parser.add_argument("--gae-lambda", type=float, default=0.99)
    parser.add_argument("--snapshot-interval", type=int, default=16)
    parser.add_argument("--opponent-pool-size", type=int, default=8)
    parser.add_argument("--current-opponent-prob", type=float, default=0.8)
    parser.add_argument("--checkpoint-dir", default="checkpoints/carl_gru_ppo")
    parser.add_argument("--tensorboard-dir")
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def policy_snapshot(policy: GRUPolicy) -> dict[str, torch.Tensor]:
    return {
        key: value.detach().cpu().clone() for key, value in policy.state_dict().items()
    }


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CARL requires a CUDA-capable PyTorch installation")
    if args.tick_skip < 1:
        raise ValueError("tick-skip must be positive")
    if args.max_ticks % (args.rollout_steps * args.tick_skip):
        raise ValueError("max-ticks must be divisible by rollout-steps * tick-skip")
    if args.rollout_steps % args.sequence_len:
        raise ValueError("rollout-steps must be divisible by sequence-len")
    if args.batch_size % args.sequence_len:
        raise ValueError("batch-size must be divisible by sequence-len")
    if args.snapshot_interval < 1 or args.opponent_pool_size < 1:
        raise ValueError("snapshot settings must be positive")
    if args.ent_coef < 0:
        raise ValueError("ent-coef cannot be negative")
    if args.reward_scale <= 0:
        raise ValueError("reward-scale must be positive")
    if not 0 <= args.team_spirit <= 1:
        raise ValueError("team-spirit must be in [0, 1]")
    if not 0 <= args.current_opponent_prob <= 1:
        raise ValueError("current-opponent-prob must be in [0, 1]")
    if not 0 < args.gamma <= 1 or not 0 < args.gae_lambda <= 1:
        raise ValueError("gamma and gae-lambda must be in (0, 1]")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    env = CarlEnv(
        args.n_sim,
        args.n_blue,
        args.n_orange,
        args.seed,
        args.max_ticks,
        args.reward_scale,
        args.tick_skip,
        args.team_spirit,
    )
    shared_backbone = GRUBackbone(env.obs_dim, args.hidden_size).to("cuda")

    def make_policy(backbone: GRUBackbone = None) -> GRUPolicy:
        return GRUPolicy(env, args.hidden_size, backbone).to("cuda")

    policy = make_policy(shared_backbone)
    opponent = make_policy()
    opponent.load_state_dict(policy.state_dict())
    opponent.requires_grad_(False).eval()
    critic = GRUValue(args.hidden_size, shared_backbone).to("cuda")

    captures = (
        LogProbCapture(),
        PolicyVersionCapture(policy),
        ValueCapture(critic),
        RecurrentStateCapture(),
    )

    rollout_buffer = RolloutBuffer(args.rollout_steps, env.n_envs, "cuda")
    optimizer = Adam(unique_parameters((policy, critic)), lr=args.learning_rate)
    ppo = PPOLearner(
        transforms=(GAE(gamma=args.gamma, lambda_=args.gae_lambda),),
        optimizer=PPOOptimizer(
            policy,
            critic,
            RecurrentRolloutMinibatches(
                args.sequence_len,
                sequences_per_batch=args.batch_size // args.sequence_len,
                epochs=args.epochs,
            ),
            OptimizerStep((policy, critic), optimizer, max_grad_norm=0.5),
            PPOConfig(
                clip=0.2,
                value_clip=0.2,
                value_coef=0.5,
                entropy_coef=args.ent_coef,
            ),
        ),
    )

    samples_per_update = env.n_envs * args.rollout_steps
    updates = args.total_timesteps // samples_per_update
    if updates < 1:
        raise ValueError(
            f"total-timesteps must be at least {samples_per_update:,} for one rollout"
        )

    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    snapshots = deque(maxlen=args.opponent_pool_size)
    snapshots.append((0, policy_snapshot(policy)))
    torch.save(snapshots[0][1], checkpoint_dir / "policy_000000.pt")

    rating_system = trueskill.TrueSkill(draw_probability=0.0)
    ratings = {0: rating_system.create_rating()}
    current_rating = rating_system.create_rating()

    self_play_stats = {
        "games": 0,
        "wins": 0,
        "draws": 0,
        "goals_for": 0,
        "goals_against": 0,
    }
    goal_stats = {"games": 0, "for": 0, "against": 0}
    recent_episode_goals = deque(maxlen=50)
    current_opponent = 0
    play_current = torch.zeros(args.n_sim, dtype=torch.bool, device="cuda")

    all_observations = env.reset()
    tensorboard_dir = args.tensorboard_dir or datetime.now().strftime(
        "runs/carl_gru_ppo/%Y%m%d-%H%M%S"
    )
    logger = Logger(log_dir=tensorboard_dir)
    logger.writer.add_text("config", json.dumps(vars(args), indent=2), 0)
    episode_return = torch.zeros(env.n_envs, device="cuda")
    episode_length = torch.zeros(env.n_envs, dtype=torch.long, device="cuda")
    match_score = torch.zeros(args.n_sim, device="cuda")
    match_goals_for = torch.zeros(args.n_sim, device="cuda")
    match_goals_against = torch.zeros(args.n_sim, device="cuda")
    policy_state = policy.initial_state(env.n_envs)
    opponent_state = opponent.initial_state(args.n_sim * args.n_orange)
    for update in logger.progress(updates):
        eligible = [
            (snapshot_id, state)
            for snapshot_id, state in snapshots
            if ratings[snapshot_id].mu > current_rating.mu - 10.0
        ]
        if not eligible:
            eligible = [
                max(
                    snapshots,
                    key=lambda item: ratings[item[0]].mu,
                )
            ]

        weights = np.asarray(
            [max(ratings[snapshot_id].mu, 1e-6) for snapshot_id, _ in eligible]
        )
        weights /= weights.sum()
        pool_index = np.random.choice(len(eligible), p=weights)
        current_opponent, state = eligible[pool_index]
        opponent.load_state_dict(state)
        opponent_state.zero_()

        play_current.copy_(
            torch.rand(args.n_sim, device="cuda") < args.current_opponent_prob
        )

        for rollout_step in range(args.rollout_steps):
            learner_observation = env.team_data(all_observations, blue=True)
            opponent_observation = env.team_data(all_observations, blue=False)
            with torch.no_grad():
                learner_output = policy.act(learner_observation, policy_state)
                opponent_output = opponent.act(opponent_observation, opponent_state)
                current_output = policy.act(opponent_observation, opponent_state)
                current_mask = play_current.repeat_interleave(args.n_orange)
                opponent_actions = torch.where(
                    current_mask[:, None],
                    current_output.action,
                    opponent_output.action,
                )
                next_opponent_state = torch.where(
                    current_mask[:, None],
                    current_output.next_state,
                    opponent_output.next_state,
                )

            actions = env.combine_actions(learner_output.action, opponent_actions)
            env_step = env.step(actions)
            learner_env_step = EnvStep(
                next_obs=env.team_data(env_step.next_obs, blue=True),
                observation=env.team_data(env_step.observation, blue=True),
                reward=env.team_data(env_step.reward, blue=True),
                terminated=env.team_data(env_step.terminated, blue=True),
                truncated=env.team_data(env_step.truncated, blue=True),
            )

            record = build_record(
                CaptureContext(
                    observation=learner_observation,
                    state=policy_state,
                    policy_output=learner_output,
                    env_step=learner_env_step,
                ),
                captures,
            )
            rollout_buffer.append(record)

            episode_return += learner_env_step.reward
            episode_length += args.tick_skip
            learner_goal = env.team_data(env_step.info["goal"], blue=True)
            learner_goal = learner_goal.view(args.n_sim, args.n_blue)[:, 0]
            match_score += learner_goal
            match_goals_for += learner_goal > 0
            match_goals_against += learner_goal < 0

            all_observations = env_step.observation
            policy_state = learner_output.next_state
            opponent_state = next_opponent_state

            learner_done = learner_env_step.done
            if learner_done.any():
                done_sim = learner_done.view(args.n_sim, args.n_blue).any(dim=1)
                ended_score = match_score[done_sim]
                ended_goals_for = match_goals_for[done_sim]
                ended_goals_against = match_goals_against[done_sim]
                games = int(done_sim.sum().item())
                global_t = (update * args.rollout_steps + rollout_step + 1) * env.n_envs
                logger.episode(
                    global_t,
                    {
                        "reward": episode_return[learner_done].cpu().tolist(),
                        "length": episode_length[learner_done].cpu().tolist(),
                    },
                )

                stats = self_play_stats
                stats["games"] += games
                stats["wins"] += (ended_score > 0).sum().item()
                stats["draws"] += (ended_score == 0).sum().item()
                episode_goals_for = int(ended_goals_for.sum().item())
                episode_goals_against = int(ended_goals_against.sum().item())
                stats["goals_for"] += episode_goals_for
                stats["goals_against"] += episode_goals_against
                goal_stats["games"] += games
                goal_stats["for"] += episode_goals_for
                goal_stats["against"] += episode_goals_against
                total_goals = goal_stats["for"] + goal_stats["against"]
                recent_episode_goals.extend(
                    (ended_goals_for + ended_goals_against).cpu().tolist()
                )

                rating_outcomes = match_score[done_sim & ~play_current].cpu()
                if len(rating_outcomes) > 32:
                    rating_outcomes = rating_outcomes[
                        torch.randperm(len(rating_outcomes))[:32]
                    ]
                for outcome in rating_outcomes.tolist():
                    if outcome > 0:
                        current_rating, ratings[current_opponent] = (
                            rating_system.rate_1vs1(
                                current_rating, ratings[current_opponent]
                            )
                        )
                    elif outcome < 0:
                        ratings[current_opponent], current_rating = (
                            rating_system.rate_1vs1(
                                ratings[current_opponent], current_rating
                            )
                        )

                logger.update(
                    {
                        "SelfPlay": {
                            "prior_opponent": current_opponent,
                            "current_fraction": play_current.float().mean().item(),
                            "games": stats["games"],
                            "win_rate": stats["wins"] / stats["games"],
                            "draw_rate": stats["draws"] / stats["games"],
                            "goal_diff": (stats["goals_for"] - stats["goals_against"])
                            / stats["games"],
                            "current_mu": current_rating.mu,
                            "opponent_mu": ratings[current_opponent].mu,
                        },
                        "Goals": {
                            "for": goal_stats["for"],
                            "against": goal_stats["against"],
                            "total": total_goals,
                            "per_episode_50": np.mean(recent_episode_goals),
                        },
                    }
                )

                episode_return[learner_done] = 0
                episode_length[learner_done] = 0
                match_score[done_sim] = 0
                match_goals_for[done_sim] = 0
                match_goals_against[done_sim] = 0
                policy_state[learner_done] = 0
                opponent_done = done_sim.repeat_interleave(args.n_orange)
                opponent_state[opponent_done] = 0

        global_t = (update + 1) * samples_per_update
        logger.update(ppo.update(rollout_buffer.finish()), step=global_t)
        rollout_buffer.clear()

        update_number = update + 1
        if update_number % args.snapshot_interval == 0:
            state = policy_snapshot(policy)
            snapshots.append((update_number, state))
            ratings[update_number] = trueskill.Rating(
                mu=current_rating.mu, sigma=current_rating.sigma
            )
            torch.save(state, checkpoint_dir / f"policy_{update_number:06d}.pt")

            rating_data = {
                str(snapshot_id): {
                    "mu": rating.mu,
                    "sigma": rating.sigma,
                    "skill": rating.mu - 3.0 * rating.sigma,
                }
                for snapshot_id, rating in ratings.items()
            }
            (checkpoint_dir / "opponent_ratings.json").write_text(
                json.dumps(rating_data, indent=2) + "\n"
            )


if __name__ == "__main__":
    main()
