import json
from pathlib import Path

import torch
import trueskill


class TrueSkillEvaluator:
    def __init__(
        self,
        policy,
        opponent_pool,
        env_factory,
        logger,
        checkpoint_dir: Path,
        interval:       int,
        num_matches:    int,
        team_sizes:     tuple[int, int],
        max_steps:      int,
        opponents:      int,
        draw_probability: float,
        seed:            int,
    ) -> None:
        if interval < 1 or num_matches < 1 or max_steps < 1 or opponents < 1:
            raise ValueError("TrueSkill settings must be positive")
        if any(size < 1 for size in team_sizes):
            raise ValueError("team sizes must be positive")
        if not 0.0 <= draw_probability < 1.0:
            raise ValueError("draw probability must be between zero and one")

        self.policy = policy
        self.opponent_pool = opponent_pool
        self.env_factory = env_factory
        self.logger = logger
        self.checkpoint_dir = checkpoint_dir
        self.interval = interval
        self.num_matches = num_matches
        self.n_blue, self.n_orange = team_sizes
        self.players_per_match = sum(team_sizes)
        self.max_steps = max_steps
        self.opponents = opponents
        self.seed = seed
        self.next_evaluation = interval
        self.current_step = 0
        self.evaluation_count = 0
        self.rating_system = trueskill.TrueSkill(
            draw_probability=draw_probability
        )
        self.current_rating = self.rating_system.create_rating()
        self.snapshot_ratings = {}
        self.rating_games = {}
        self.env = None

    def ready(self, step: int) -> bool:
        self.current_step = step
        if step < self.next_evaluation:
            return False
        while self.next_evaluation <= step:
            self.next_evaluation += self.interval
        return True

    @torch.no_grad()
    def run(self) -> None:
        if self.env is None:
            self.env = self.env_factory()
            expected = self.num_matches * self.players_per_match
            if self.env.n_envs != expected:
                raise ValueError(
                    f"evaluation environment has {self.env.n_envs} actors, "
                    f"expected {expected}"
                )

        snapshot_ids = self.opponent_pool.select_ids(self.opponents)
        wins = draws = games = 0
        devices = [self.policy.device.index or 0]
        with torch.random.fork_rng(devices=devices):
            torch.manual_seed(self.seed + self.evaluation_count)
            for snapshot_id in snapshot_ids:
                opponent = self.opponent_pool.policy(
                    snapshot_id, self.policy.device
                )
                outcomes = torch.cat(
                    (
                        self._play(opponent, current_is_blue=True),
                        self._play(opponent, current_is_blue=False),
                    )
                ).cpu()
                wins += int(outcomes.gt(0).sum())
                draws += int(outcomes.eq(0).sum())
                games += len(outcomes)
                self._rate(snapshot_id, outcomes)

        self.evaluation_count += 1
        rating = self.current_rating
        self.logger.update(
            {
                "TrueSkill": {
                    "mu":        rating.mu,
                    "sigma":     rating.sigma,
                    "skill":     rating.mu - 3.0 * rating.sigma,
                    "games":      games,
                    "win_rate":   wins / games,
                    "draw_rate":  draws / games,
                    "opponents":  len(snapshot_ids),
                }
            },
            step=self.current_step,
        )
        self._write_ratings()

    def close(self) -> None:
        if self.env is not None:
            self.env.close()
            self.env = None

    def _play(self, opponent, current_is_blue: bool) -> torch.Tensor:
        observation = self.env.reset()
        current_state = self.policy.initial_state(
            self.num_matches
            * (self.n_blue if current_is_blue else self.n_orange)
        )
        opponent_state = opponent.initial_state(
            self.num_matches
            * (self.n_orange if current_is_blue else self.n_blue)
        )
        active = torch.ones(
            self.num_matches, dtype=torch.bool, device=self.policy.device
        )
        outcomes = torch.zeros(
            self.num_matches, dtype=torch.float32, device=self.policy.device
        )

        for _ in range(self.max_steps):
            grouped = observation.view(
                self.num_matches, self.players_per_match, -1
            )
            blue = grouped[:, : self.n_blue].flatten(0, 1)
            orange = grouped[:, self.n_blue :].flatten(0, 1)
            if current_is_blue:
                blue_output = self.policy.act(blue, current_state)
                orange_output = opponent.act(orange, opponent_state)
                current_state = blue_output.next_state
                opponent_state = orange_output.next_state
            else:
                blue_output = opponent.act(blue, opponent_state)
                orange_output = self.policy.act(orange, current_state)
                opponent_state = blue_output.next_state
                current_state = orange_output.next_state

            action = torch.cat(
                (
                    blue_output.action.view(
                        self.num_matches, self.n_blue, -1
                    ),
                    orange_output.action.view(
                        self.num_matches, self.n_orange, -1
                    ),
                ),
                dim=1,
            ).flatten(0, 1)
            observation, reward, terminated, truncated, _ = self.env.step(action)
            done = (terminated | truncated).view(
                self.num_matches, self.players_per_match
            ).any(dim=-1)
            finished = active & done
            blue_result = reward.view(
                self.num_matches, self.players_per_match
            )[:, 0]
            outcomes[finished] = (
                blue_result[finished]
                if current_is_blue
                else -blue_result[finished]
            )
            active &= ~done
            if not active.any():
                break

        return outcomes

    def _rate(self, snapshot_id: int, outcomes: torch.Tensor) -> None:
        opponent_rating = self.snapshot_ratings.setdefault(
            snapshot_id,
            trueskill.Rating(
                mu=self.current_rating.mu,
                sigma=self.current_rating.sigma,
            ),
        )
        rated_games = 0
        for outcome in outcomes.tolist():
            if outcome > 0:
                self.current_rating, opponent_rating = self.rating_system.rate_1vs1(
                    self.current_rating, opponent_rating
                )
                rated_games += 1
            elif outcome < 0:
                opponent_rating, self.current_rating = self.rating_system.rate_1vs1(
                    opponent_rating, self.current_rating
                )
                rated_games += 1
        self.snapshot_ratings[snapshot_id] = opponent_rating
        self.rating_games[snapshot_id] = (
            self.rating_games.get(snapshot_id, 0) + rated_games
        )

    def _write_ratings(self) -> None:
        snapshots = {
            str(snapshot_id): {
                "mu":    rating.mu,
                "sigma": rating.sigma,
                "skill": rating.mu - 3.0 * rating.sigma,
                "games": self.rating_games.get(snapshot_id, 0),
                "timesteps": self.opponent_pool.timesteps(snapshot_id),
            }
            for snapshot_id, rating in self.snapshot_ratings.items()
            if snapshot_id in self.opponent_pool.ids
        }
        payload = {
            "current": {
                "mu":    self.current_rating.mu,
                "sigma": self.current_rating.sigma,
                "skill": self.current_rating.mu - 3.0 * self.current_rating.sigma,
            },
            "snapshots": snapshots,
        }
        (self.checkpoint_dir / "trueskill_ratings.json").write_text(
            json.dumps(payload, indent=2) + "\n"
        )


__all__ = ["TrueSkillEvaluator"]
