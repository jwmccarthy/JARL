#!/usr/bin/env python3
"""Train PPO with JARL on the CUDA-accelerated CARL environment."""

import argparse
from collections import deque
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

from jarl.data.multi import MultiTensor
from jarl.envs.space import torch_space
from jarl.log.logger import Logger
from jarl.modules.utils import init_layer
from jarl.train.graph import TrainGraph
from jarl.train.modify.compute import (
    ComputeAdvantages,
    ComputeReturns,
)
from jarl.train.optim import Optimizer
from jarl.train.sample.sequence import SequenceSampler
from jarl.train.update.base import GradientUpdate


class CarlEnv:
    """Small GPU-native adapter implementing the environment API JARL needs."""

    def __init__(
        self,
        n_sim: int,
        n_blue: int,
        n_orange: int,
        seed: int,
        max_ticks: int,
        goal_reward_scale: float,
        ball_speed_reward_scale: float,
        ball_touch_reward_scale: float,
        ball_goal_velocity_reward_scale: float,
        ball_distance_reward_scale: float,
        reward_scale: float,
        tick_skip: int,
    ) -> None:
        self.env = carl.Env(n_sim, n_blue, n_orange, seed)
        self.env.max_ticks = max_ticks
        self.n_sim = n_sim
        self.n_blue = n_blue
        self.n_orange = n_orange
        self.n_cars = self.env.n_cars
        self.n_agents = n_sim * self.n_cars
        self.n_envs = n_sim * n_blue
        self.max_ticks = max_ticks
        self.goal_reward_scale = goal_reward_scale
        self.ball_speed_reward_scale = ball_speed_reward_scale
        self.ball_touch_reward_scale = ball_touch_reward_scale
        self.ball_goal_velocity_reward_scale = ball_goal_velocity_reward_scale
        self.ball_distance_reward_scale = ball_distance_reward_scale
        self.reward_scale = reward_scale
        self.tick_skip = tick_skip
        self.elapsed_ticks = 0
        self.score = torch.zeros(n_sim, device="cuda")
        self.touching = torch.zeros(
            (n_sim, self.n_cars), dtype=torch.bool, device="cuda"
        )
        self.ball_distance = torch.zeros(
            (n_sim, self.n_cars), device="cuda"
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

        obs = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.env.obs_dim,), dtype=np.float32
        )
        act = gym.spaces.MultiDiscrete(
            np.asarray(self.env.action_nvec[0], dtype=np.int32), dtype=np.int32
        )
        self.obs_space = torch_space(obs, device="cuda")
        self.act_space = torch_space(act, device="cuda")

    def _obs(self, source=None) -> torch.Tensor:
        if source is None:
            source = self.env.get_obs()
        raw = (
            source.clone()
            if isinstance(source, torch.Tensor)
            else torch.from_dlpack(source).clone()
        )
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

        # CARL observations are raw physics units. Scale each documented field
        # to keep the MLP inputs in a useful range.
        ball[..., 0:3] /= 6000.0
        ball[..., 3:6] /= 6000.0
        ball[..., 6:9] /= 6.0
        cars[..., 0:3] /= 6000.0
        cars[..., 3:6] /= 2300.0
        cars[..., 6:9] /= 6.0
        cars[..., 15] /= 100.0
        return torch.cat((ball, cars.flatten(2)), dim=-1).flatten(0, 1)

    def _ball_distance(self, raw: torch.Tensor) -> torch.Tensor:
        ball = raw[:, None, :3]
        cars = raw[:, 9:].view(self.n_sim, self.n_cars, 21)[..., :3]
        return (cars - ball).norm(dim=-1)

    def reset(self) -> torch.Tensor:
        self.env.reset()
        self.elapsed_ticks = 0
        self.score.zero_()
        self.touching.zero_()
        raw = torch.from_dlpack(self.env.get_obs()).clone()
        self.ball_distance.copy_(self._ball_distance(raw))
        return self._obs(raw)

    def step(self, actions: torch.Tensor):
        actions = actions.to(dtype=torch.int32).contiguous()
        carl_actions = actions.view(self.n_sim, self.n_cars, -1)
        touch = torch.zeros_like(self.touching)
        for _ in range(self.tick_skip):
            raw_obs = torch.from_dlpack(self.env.step(carl_actions)).clone()
            touching = torch.from_dlpack(
                self.env.get_ball_touches()
            ).bool().clone()
            touch |= touching & ~self.touching
            self.touching.copy_(touching)
        next_obs = self._obs(raw_obs)
        score = torch.from_dlpack(self.env.get_rewards()).clone()
        goal = ((score - self.score)[:, None] * self.team_sign).flatten()
        touch = touch.float().flatten()
        ball_distance = self._ball_distance(raw_obs)
        distance_progress = (
            (self.ball_distance - ball_distance).flatten() / 6000.0
        )
        self.ball_distance.copy_(ball_distance)

        ball_speed = raw_obs[:, 3:6].norm(dim=-1) / 6000.0
        speed_reward = (
            ball_speed[:, None]
            .expand(-1, self.n_cars)
            .flatten()
            * self.ball_speed_reward_scale
            * self.tick_skip
        )
        target_y = self.team_sign[None, :] * 5124.25
        to_goal = torch.stack(
            (
                -raw_obs[:, None, 0].expand(-1, self.n_cars),
                target_y - raw_obs[:, None, 1],
            ),
            dim=-1,
        )
        to_goal = torch.nn.functional.normalize(to_goal, dim=-1)
        goal_velocity = (
            raw_obs[:, None, 3:5] * to_goal
        ).sum(-1).flatten() / 6000.0
        reward = (
            goal * self.goal_reward_scale
            + speed_reward
            + touch * self.ball_touch_reward_scale
            + goal_velocity
            * self.ball_goal_velocity_reward_scale
            * self.tick_skip
            + distance_progress * self.ball_distance_reward_scale
        ) * self.reward_scale
        self.score.copy_(score)

        self.elapsed_ticks += self.tick_skip
        truncated = self.elapsed_ticks >= self.max_ticks
        done = torch.full(
            (self.n_agents,), truncated, dtype=torch.bool, device="cuda"
        )
        transition = {
            "act": actions,
            "rew": reward,
            "goal": goal,
            "don": done,
            "trc": done.clone(),
            "nxt": next_obs,
        }

        if truncated:
            current_obs = self.reset()
        else:
            current_obs = next_obs
        return transition, current_obs

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
    def __init__(self, obs_dim: int, hidden_size: int) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.encoder = nn.Sequential(
            init_layer(nn.Linear(obs_dim, hidden_size)), nn.ReLU()
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
        self, obs: torch.Tensor, hidden: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.gru(self.encoder(obs), hidden)


class GRUPolicy(nn.Module):
    def __init__(self, env: CarlEnv, hidden_size: int) -> None:
        super().__init__()
        self.backbone = GRUBackbone(env.env.obs_dim, hidden_size)
        nvec = env.act_space.nvec.flatten().tolist()
        self.sizes = tuple(int(value) for value in nvec)
        self.action_shape = env.act_space.shape
        self.output = init_layer(
            nn.Linear(hidden_size, sum(self.sizes)), std=0.01
        )

    def initial_state(self, batch_size: int) -> torch.Tensor:
        return self.backbone.initial_state(batch_size, self.output.weight.device)

    def _dist(
        self, obs: torch.Tensor, hidden: torch.Tensor
    ) -> tuple[list[Categorical], torch.Tensor]:
        features, next_hidden = self.backbone(obs, hidden)
        logits = self.output(features).split(self.sizes, dim=-1)
        return [Categorical(logits=value) for value in logits], next_hidden

    def step(
        self, obs: torch.Tensor, hidden: torch.Tensor, sample: bool = True
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        distributions, next_hidden = self._dist(obs.unsqueeze(0), hidden)
        if sample:
            action = torch.stack([dist.sample() for dist in distributions], -1)
        else:
            action = torch.stack(
                [dist.logits.argmax(dim=-1) for dist in distributions], -1
            )
        action = action.reshape(*action.shape[:-1], *self.action_shape)
        logprob = torch.stack(
            [dist.log_prob(action[..., i]) for i, dist in enumerate(distributions)],
            dim=-1,
        ).sum(-1)
        return action.squeeze(0), logprob.squeeze(0), next_hidden

    def evaluate(
        self, obs: torch.Tensor, act: torch.Tensor, hidden: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        distributions, _ = self._dist(obs, hidden)
        flat_act = act.reshape(*act.shape[:-len(self.action_shape)], -1)
        logprob = torch.stack(
            [
                dist.log_prob(flat_act[..., i])
                for i, dist in enumerate(distributions)
            ],
            dim=-1,
        ).sum(-1)
        entropy = torch.stack(
            [dist.entropy() for dist in distributions], dim=-1
        ).sum(-1)
        return logprob, entropy


class GRUValue(nn.Module):
    def __init__(self, obs_dim: int, hidden_size: int) -> None:
        super().__init__()
        self.backbone = GRUBackbone(obs_dim, hidden_size)
        self.output = init_layer(nn.Linear(hidden_size, 1), std=1.0)

    def initial_state(self, batch_size: int) -> torch.Tensor:
        return self.backbone.initial_state(batch_size, self.output.weight.device)

    def step(
        self, obs: torch.Tensor, hidden: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        features, next_hidden = self.backbone(obs.unsqueeze(0), hidden)
        return self.output(features).squeeze(0).squeeze(-1), next_hidden

    def evaluate(self, obs: torch.Tensor, hidden: torch.Tensor) -> torch.Tensor:
        features, _ = self.backbone(obs, hidden)
        return self.output(features).squeeze(-1)


class RecurrentPPOUpdate(GradientUpdate):
    _requires_keys = {
        "obs", "act", "adv", "lgp", "val", "ret", "policy_h", "critic_h"
    }

    def __init__(
        self,
        freq: int,
        policy: GRUPolicy,
        critic: GRUValue,
        optimizer: Optimizer,
        clip: float,
        val_coef: float,
        ent_coef: float,
    ) -> None:
        super().__init__(freq, [policy, critic], optimizer=optimizer)
        self.policy = policy
        self.critic = critic
        self.clip = clip
        self.val_coef = val_coef
        self.ent_coef = ent_coef

    def loss(self, data: MultiTensor):
        policy_h = data.policy_h[0].unsqueeze(0).contiguous()
        critic_h = data.critic_h[0].unsqueeze(0).contiguous()
        logprob, entropy = self.policy.evaluate(data.obs, data.act, policy_h)
        value = self.critic.evaluate(data.obs, critic_h)

        logratio = logprob - data.lgp
        ratio = logratio.exp()
        with torch.no_grad():
            approx_kl = ((ratio - 1) - logratio).mean().item()
        advantage = (data.adv - data.adv.mean()) / (data.adv.std() + 1e-8)
        policy_loss = -torch.min(
            advantage * ratio,
            advantage * torch.clamp(ratio, 1 - self.clip, 1 + self.clip),
        ).mean()
        entropy_loss = self.ent_coef * entropy.mean()

        value_loss = (value - data.ret).pow(2)
        clipped_value = data.val + torch.clamp(
            value - data.val, -self.clip, self.clip
        )
        value_loss = 0.5 * torch.max(
            value_loss, (clipped_value - data.ret).pow(2)
        )
        value_loss = self.val_coef * value_loss.mean()
        loss = policy_loss - entropy_loss + value_loss
        return loss, {
            "policy_loss": policy_loss.item(),
            "entropy": entropy.mean().item(),
            "entropy_loss": entropy_loss.item(),
            "approx_kl": approx_kl,
            "critic_loss": value_loss.item(),
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--total-timesteps", type=int, default=1_000_000_000)
    parser.add_argument("--n-sim", type=int, default=1024)
    parser.add_argument("--n-blue", type=int, default=1)
    parser.add_argument("--n-orange", type=int, default=1)
    parser.add_argument("--max-ticks", type=int, default=4096)
    parser.add_argument("--tick-skip", type=int, default=4)
    parser.add_argument("--rollout-steps", type=int, default=512)
    parser.add_argument("--sequence-len", type=int, default=32)
    parser.add_argument("--hidden-size", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=16384)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=2.5e-4)
    parser.add_argument("--ent-coef", type=float, default=1e-3)
    parser.add_argument("--goal-reward-scale", type=float, default=10.0)
    parser.add_argument("--ball-speed-reward-scale", type=float, default=1e-3)
    parser.add_argument("--ball-touch-reward-scale", type=float, default=0.1)
    parser.add_argument(
        "--ball-goal-velocity-reward-scale", type=float, default=1e-3
    )
    parser.add_argument("--ball-distance-reward-scale", type=float, default=0.05)
    parser.add_argument("--reward-scale", type=float, default=1.0)
    parser.add_argument("--gamma", type=float, default=0.999)
    parser.add_argument("--gae-lambda", type=float, default=0.99)
    parser.add_argument("--snapshot-interval", type=int, default=50)
    parser.add_argument("--opponent-pool-size", type=int, default=8)
    parser.add_argument("--trueskill-n-sim", type=int, default=64)
    parser.add_argument("--trueskill-opponents", type=int, default=3)
    parser.add_argument("--trueskill-draw-probability", type=float, default=0.98)
    parser.add_argument("--checkpoint-dir", default="checkpoints/carl_gru_ppo")
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def policy_snapshot(policy: GRUPolicy) -> dict[str, torch.Tensor]:
    return {
        key: value.detach().cpu().clone()
        for key, value in policy.state_dict().items()
    }


@torch.no_grad()
def play_matches(
    env: CarlEnv,
    blue_policy: GRUPolicy,
    orange_policy: GRUPolicy,
) -> torch.Tensor:
    obs = env.reset()
    score = torch.zeros(env.n_sim, device="cuda")
    blue_h = blue_policy.initial_state(env.n_sim * env.n_blue)
    orange_h = orange_policy.initial_state(env.n_sim * env.n_orange)
    for _ in range(env.max_ticks // env.tick_skip):
        blue_obs = env.team_data(obs, blue=True)
        orange_obs = env.team_data(obs, blue=False)
        blue_actions, _, blue_h = blue_policy.step(blue_obs, blue_h)
        orange_actions, _, orange_h = orange_policy.step(orange_obs, orange_h)
        actions = env.combine_actions(blue_actions, orange_actions)
        transition, obs = env.step(actions)
        blue_goal = env.team_data(transition["goal"], blue=True)
        score += blue_goal.view(env.n_sim, env.n_blue)[:, 0]
    return score.cpu()


def select_rating_opponents(
    ratings: dict[int, trueskill.Rating], count: int
) -> list[int]:
    ids = list(ratings)
    selected = []
    for snapshot_id in (
        0,
        max(ids),
        max(ids, key=lambda key: ratings[key].mu - 3 * ratings[key].sigma),
    ):
        if snapshot_id not in selected:
            selected.append(snapshot_id)
    for snapshot_id in reversed(ids):
        if snapshot_id not in selected:
            selected.append(snapshot_id)
    return selected[:count]


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CARL requires a CUDA-capable PyTorch installation")
    if args.tick_skip < 1:
        raise ValueError("tick-skip must be positive")
    if args.max_ticks % (args.rollout_steps * args.tick_skip):
        raise ValueError(
            "max-ticks must be divisible by rollout-steps * tick-skip"
        )
    if args.rollout_steps % args.sequence_len:
        raise ValueError("rollout-steps must be divisible by sequence-len")
    if args.batch_size % args.sequence_len:
        raise ValueError("batch-size must be divisible by sequence-len")
    if args.snapshot_interval < 1 or args.opponent_pool_size < 1:
        raise ValueError("snapshot settings must be positive")
    if args.trueskill_n_sim < 1 or args.trueskill_opponents < 1:
        raise ValueError("TrueSkill settings must be positive")
    if not 0 <= args.trueskill_draw_probability < 1:
        raise ValueError("trueskill-draw-probability must be in [0, 1)")
    if args.ent_coef < 0:
        raise ValueError("ent-coef cannot be negative")
    if args.goal_reward_scale <= 0 or args.reward_scale <= 0:
        raise ValueError("goal-reward-scale and reward-scale must be positive")
    if min(
        args.ball_speed_reward_scale,
        args.ball_touch_reward_scale,
        args.ball_goal_velocity_reward_scale,
        args.ball_distance_reward_scale,
    ) < 0:
        raise ValueError("shaping reward scales cannot be negative")
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
        args.goal_reward_scale,
        args.ball_speed_reward_scale,
        args.ball_touch_reward_scale,
        args.ball_goal_velocity_reward_scale,
        args.ball_distance_reward_scale,
        args.reward_scale,
        args.tick_skip,
    )
    rating_match_env = CarlEnv(
        args.trueskill_n_sim,
        args.n_blue,
        args.n_orange,
        args.seed + 1,
        args.max_ticks,
        args.goal_reward_scale,
        args.ball_speed_reward_scale,
        args.ball_touch_reward_scale,
        args.ball_goal_velocity_reward_scale,
        args.ball_distance_reward_scale,
        args.reward_scale,
        args.tick_skip,
    )

    def make_policy() -> GRUPolicy:
        return GRUPolicy(env, args.hidden_size).to("cuda")

    policy = make_policy()
    opponent = make_policy()
    opponent.load_state_dict(policy.state_dict())
    opponent.requires_grad_(False).eval()
    rating_opponent = make_policy()
    rating_opponent.load_state_dict(policy.state_dict())
    rating_opponent.requires_grad_(False).eval()
    critic = GRUValue(env.env.obs_dim, args.hidden_size).to("cuda")

    ppo = (
        TrainGraph(
            RecurrentPPOUpdate(
                args.rollout_steps,
                policy,
                critic,
                clip=0.2,
                val_coef=0.5,
                ent_coef=args.ent_coef,
                optimizer=Optimizer(Adam, lr=args.learning_rate),
            ),
            SequenceSampler(
                args.sequence_len,
                batch_len=args.batch_size // args.sequence_len,
                num_epoch=args.epochs,
            ),
        )
        .add_modifier(ComputeAdvantages(gamma=args.gamma, lmbda=args.gae_lambda))
        .add_modifier(ComputeReturns())
        .compile()
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
    snapshot_states = {0: snapshots[0][1]}
    rating_system = trueskill.TrueSkill(
        draw_probability=args.trueskill_draw_probability
    )
    ratings = {0: rating_system.create_rating()}
    rating_games = {0: 0}
    opponent_stats = {}
    goal_stats = {"games": 0, "for": 0, "against": 0}
    current_opponent = 0

    all_obs = env.reset()
    logger = Logger()
    episode_return = torch.zeros(env.n_envs, device="cuda")
    match_score = torch.zeros(args.n_sim, device="cuda")
    match_goals_for = 0
    match_goals_against = 0
    policy_h = policy.initial_state(env.n_envs)
    critic_h = critic.initial_state(env.n_envs)
    opponent_h = opponent.initial_state(args.n_sim * args.n_orange)
    for update in logger.progress(updates):
        if env.elapsed_ticks == 0:
            pool_index = np.random.randint(len(snapshots))
            current_opponent, state = snapshots[pool_index]
            opponent.load_state_dict(state)

        rollout = {
            key: []
            for key in (
                "obs", "act", "rew", "don", "trc", "nxt",
                "lgp", "val", "next_val", "policy_h", "critic_h",
            )
        }
        for rollout_step in range(args.rollout_steps):
            learner_obs = env.team_data(all_obs, blue=True)
            opponent_obs = env.team_data(all_obs, blue=False)
            with torch.no_grad():
                learner_actions, logprob, next_policy_h = policy.step(
                    learner_obs, policy_h
                )
                opponent_actions, _, next_opponent_h = opponent.step(
                    opponent_obs, opponent_h
                )
                value, next_critic_h = critic.step(learner_obs, critic_h)
            actions = env.combine_actions(learner_actions, opponent_actions)
            transition, next_obs = env.step(actions)
            learner_transition = {
                key: env.team_data(value, blue=True)
                for key, value in transition.items()
                if key not in {"act", "goal"}
            }
            learner_transition["act"] = learner_actions
            learner_next_obs = env.team_data(transition["nxt"], blue=True)
            with torch.no_grad():
                next_value, _ = critic.step(learner_next_obs, next_critic_h)
            rollout["obs"].append(learner_obs)
            rollout["lgp"].append(logprob)
            rollout["val"].append(value)
            rollout["next_val"].append(next_value)
            rollout["policy_h"].append(policy_h.squeeze(0))
            rollout["critic_h"].append(critic_h.squeeze(0))
            for key, value in learner_transition.items():
                rollout[key].append(value)
            episode_return += learner_transition["rew"]
            learner_goal = env.team_data(transition["goal"], blue=True)
            learner_goal = learner_goal.view(args.n_sim, args.n_blue)[:, 0]
            match_score += learner_goal
            match_goals_for += (learner_goal > 0).sum().item()
            match_goals_against += (learner_goal < 0).sum().item()
            all_obs = next_obs
            policy_h = next_policy_h
            critic_h = next_critic_h
            opponent_h = next_opponent_h

            if learner_transition["don"].any():
                global_t = (
                    update * args.rollout_steps + rollout_step + 1
                ) * env.n_envs
                logger.episode(
                    global_t,
                    {
                        "reward": episode_return.cpu().tolist(),
                        "length": [args.max_ticks] * env.n_envs,
                    },
                )
                stats = opponent_stats.setdefault(
                    current_opponent,
                    {
                        "games": 0,
                        "wins": 0,
                        "draws": 0,
                        "goals_for": 0,
                        "goals_against": 0,
                    },
                )
                stats["games"] += args.n_sim
                stats["wins"] += (match_score > 0).sum().item()
                stats["draws"] += (match_score == 0).sum().item()
                stats["goals_for"] += match_goals_for
                stats["goals_against"] += match_goals_against
                goal_stats["games"] += args.n_sim
                goal_stats["for"] += match_goals_for
                goal_stats["against"] += match_goals_against
                total_goals = goal_stats["for"] + goal_stats["against"]
                logger.update(
                    {
                        "SelfPlay": {
                            "opponent": current_opponent,
                            "games": stats["games"],
                            "win_rate": stats["wins"] / stats["games"],
                            "draw_rate": stats["draws"] / stats["games"],
                            "goal_diff": (
                                stats["goals_for"] - stats["goals_against"]
                            ) / stats["games"],
                        },
                        "Goals": {
                            "for": goal_stats["for"],
                            "against": goal_stats["against"],
                            "total": total_goals,
                            "per_game": total_goals / goal_stats["games"],
                        }
                    }
                )
                episode_return.zero_()
                match_score.zero_()
                match_goals_for = 0
                match_goals_against = 0
                policy_h.zero_()
                critic_h.zero_()
                opponent_h.zero_()

        data = MultiTensor(
            **{key: torch.stack(values) for key, values in rollout.items()}
        )
        ppo.ready(args.rollout_steps)
        logger.update(ppo.update(data))

        update_number = update + 1
        if update_number % args.snapshot_interval == 0:
            state = policy_snapshot(policy)
            snapshots.append((update_number, state))
            snapshot_states[update_number] = state
            torch.save(
                state, checkpoint_dir / f"policy_{update_number:06d}.pt"
            )
            ratings[update_number] = rating_system.create_rating()
            rating_games[update_number] = 0
            rating_opponents = select_rating_opponents(
                {key: value for key, value in ratings.items() if key != update_number},
                args.trueskill_opponents,
            )
            eval_wins = eval_draws = eval_games = 0
            for opponent_id in rating_opponents:
                rating_opponent.load_state_dict(snapshot_states[opponent_id])
                outcomes = torch.cat(
                    (
                        play_matches(rating_match_env, policy, rating_opponent),
                        -play_matches(rating_match_env, rating_opponent, policy),
                    )
                )
                eval_wins += (outcomes > 0).sum().item()
                eval_draws += (outcomes == 0).sum().item()
                eval_games += len(outcomes)
                for outcome in outcomes.tolist():
                    if outcome > 0:
                        ratings[update_number], ratings[opponent_id] = (
                            rating_system.rate_1vs1(
                                ratings[update_number], ratings[opponent_id]
                            )
                        )
                    elif outcome < 0:
                        ratings[opponent_id], ratings[update_number] = (
                            rating_system.rate_1vs1(
                                ratings[opponent_id], ratings[update_number]
                            )
                        )
                    else:
                        ratings[update_number], ratings[opponent_id] = (
                            rating_system.rate_1vs1(
                                ratings[update_number],
                                ratings[opponent_id],
                                drawn=True,
                            )
                        )
                rating_games[update_number] += len(outcomes)
                rating_games[opponent_id] += len(outcomes)

            rating = ratings[update_number]
            logger.update(
                {
                    "TrueSkill": {
                        "snapshot": update_number,
                        "mu": rating.mu,
                        "sigma": rating.sigma,
                        "skill": rating.mu - 3 * rating.sigma,
                        "games": rating_games[update_number],
                        "win_rate": eval_wins / eval_games,
                        "draw_rate": eval_draws / eval_games,
                    }
                }
            )
            leaderboard = {
                str(snapshot_id): {
                    "mu": value.mu,
                    "sigma": value.sigma,
                    "skill": value.mu - 3 * value.sigma,
                    "games": rating_games[snapshot_id],
                }
                for snapshot_id, value in ratings.items()
            }
            (checkpoint_dir / "trueskill_ratings.json").write_text(
                json.dumps(leaderboard, indent=2) + "\n"
            )


if __name__ == "__main__":
    main()
