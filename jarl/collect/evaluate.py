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
        checkpoint_dir:   Path,
        interval:         int,
        num_matches:      int,
        team_sizes:       tuple[int, int],
        max_steps:        int,
        opponents:        int,
        draw_probability: float,
        seed:             int,
    ) -> None:
        if interval < 1 or num_matches < 1 or max_steps < 1 or opponents < 1:
            raise ValueError("TrueSkill settings must be positive")
        if any(size < 1 for size in team_sizes):
            raise ValueError("team sizes must be positive")
        if not 0.0 <= draw_probability < 1.0:
            raise ValueError("draw probability must be between zero and one")
        if opponent_pool.checkpoint_dir is None:
            raise ValueError("TrueSkill evaluation requires persistent snapshots")
        if opponent_pool.checkpoint_dir.resolve() != checkpoint_dir.resolve():
            raise ValueError(
                "TrueSkill history and snapshots must share a checkpoint directory"
            )

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
        self.last_snapshot_id = None
        self.rating_system = trueskill.TrueSkill(
            draw_probability=draw_probability
        )
        self.history_path = checkpoint_dir / "trueskill_history.json"
        self.history = self._load_history()
        self.snapshot_ratings = {}
        self.rating_games = {}
        self.last_evaluated = {}
        self.env = None
        self._recompute_ratings()

    def ready(self, step: int) -> bool:
        self.current_step = step
        snapshot_ids = self.opponent_pool.archive_ids

        if len(snapshot_ids) < 2:
            return False

        ready = (
            snapshot_ids[-1] != self.last_snapshot_id
            or step >= self.next_evaluation
        )
        if not ready:
            return False

        while self.next_evaluation <= step:
            self.next_evaluation += self.interval

        return True

    @torch.no_grad()
    def run(self) -> None:
        snapshot_ids = self.opponent_pool.archive_ids
        if len(snapshot_ids) < 2:
            return

        if self.env is None:
            self.env = self.env_factory()
            expected = self.num_matches * self.players_per_match

            if self.env.n_envs != expected:
                raise ValueError(
                    f"evaluation environment has {self.env.n_envs} actors, "
                    f"expected {expected}"
                )

        latest_id = snapshot_ids[-1]
        opponent_ids = self._select_opponents(latest_id)
        latest = self.opponent_pool.policy(latest_id, self.policy.device)

        wins = draws = games = 0
        policy_device = torch.device(self.policy.device)
        devices = (
            [policy_device.index or 0]
            if policy_device.type == "cuda"
            else []
        )

        with torch.random.fork_rng(devices=devices):
            torch.manual_seed(self.seed + self.evaluation_count)
            for opponent_id in opponent_ids:
                opponent = self.opponent_pool.policy(
                    opponent_id, self.policy.device
                )

                outcomes = torch.cat(
                    (
                        self._play(latest, opponent),
                        -self._play(opponent, latest),
                    )
                ).cpu()

                self.history.append(
                    {
                        "step":     self.current_step,
                        "left":     latest_id,
                        "right":    opponent_id,
                        "outcomes": outcomes.sign().to(torch.int8).tolist(),
                    }
                )

                self.last_evaluated[latest_id] = self.current_step
                self.last_evaluated[opponent_id] = self.current_step
                wins += int(outcomes.gt(0).sum())
                draws += int(outcomes.eq(0).sum())
                games += len(outcomes)

        self.evaluation_count += 1
        self.last_snapshot_id = latest_id
        self._recompute_ratings()
        rating = self.snapshot_ratings[latest_id]
        self.logger.update(
            {
                "TrueSkill": {
                    "mu":          rating.mu,
                    "sigma":       rating.sigma,
                    "skill":       rating.mu - 3.0 * rating.sigma,
                    "games":       games,
                    "total_games": self.rating_games[latest_id],
                    "win_rate":    wins / games,
                    "draw_rate":   draws / games,
                    "opponents":   len(opponent_ids),
                    "snapshot_id": latest_id,
                    "timesteps":   self.opponent_pool.timesteps(latest_id),
                }
            },
            step=self.current_step,
        )
        self._write_history()
        self._write_ratings(latest_id)

    def close(self) -> None:
        if self.env is not None:
            self.env.close()
            self.env = None

    def _select_opponents(self, latest_id: int) -> tuple[int, ...]:
        candidates = [
            snapshot_id
            for snapshot_id in self.opponent_pool.archive_ids
            if snapshot_id != latest_id
        ]
        count = min(self.opponents, len(candidates))
        if not count:
            return ()

        selected = [candidates[0]]
        if count > 1 and candidates[-1] not in selected:
            selected.append(candidates[-1])

        while len(selected) < count:
            remaining = [
                snapshot_id
                for snapshot_id in candidates
                if snapshot_id not in selected
            ]
            selected.append(
                min(
                    remaining,
                    key=lambda snapshot_id: (
                        self.last_evaluated.get(snapshot_id, -1),
                        self.rating_games.get(snapshot_id, 0),
                        snapshot_id,
                    ),
                )
            )
        return tuple(selected)

    def _play(self, blue_policy, orange_policy) -> torch.Tensor:
        observation = self.env.reset()
        blue_state = blue_policy.initial_state(self.num_matches * self.n_blue)
        orange_state = orange_policy.initial_state(
            self.num_matches * self.n_orange
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
            blue_output = blue_policy.act(blue, blue_state)
            orange_output = orange_policy.act(orange, orange_state)
            blue_state = blue_output.next_state
            orange_state = orange_output.next_state

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

            outcomes[finished] = blue_result[finished]
            active &= ~done

            if not active.any():
                break

        return outcomes

    def _recompute_ratings(self) -> None:
        self.snapshot_ratings = {
            snapshot_id: self.rating_system.create_rating()
            for snapshot_id in self.opponent_pool.archive_ids
        }
        self.rating_games = {
            snapshot_id: 0 for snapshot_id in self.opponent_pool.archive_ids
        }
        self.last_evaluated = {}

        for match in self.history:
            left_id = int(match["left"])
            right_id = int(match["right"])

            if (
                left_id not in self.snapshot_ratings
                or right_id not in self.snapshot_ratings
            ):
                continue

            left = self.snapshot_ratings[left_id]
            right = self.snapshot_ratings[right_id]

            for outcome in match["outcomes"]:
                if outcome > 0:
                    left, right = self.rating_system.rate_1vs1(left, right)
                elif outcome < 0:
                    right, left = self.rating_system.rate_1vs1(right, left)
                else:
                    left, right = self.rating_system.rate_1vs1(
                        left, right, drawn=True
                    )
                self.rating_games[left_id] += 1
                self.rating_games[right_id] += 1

            self.snapshot_ratings[left_id] = left
            self.snapshot_ratings[right_id] = right

            step = int(match["step"])
            self.last_evaluated[left_id] = step
            self.last_evaluated[right_id] = step

    def _load_history(self) -> list[dict]:
        if not self.history_path.is_file():
            return []

        history = json.loads(self.history_path.read_text(encoding="utf-8"))
        if not isinstance(history, list):
            raise ValueError("TrueSkill history must be a list")

        return history

    def _write_history(self) -> None:
        self._write_json(self.history_path, self.history)

    def _write_ratings(self, latest_id: int) -> None:
        snapshots = {
            str(snapshot_id): {
                "mu":             rating.mu,
                "sigma":          rating.sigma,
                "skill":          rating.mu - 3.0 * rating.sigma,
                "games":          self.rating_games[snapshot_id],
                "timesteps":      self.opponent_pool.timesteps(snapshot_id),
                "last_evaluated": self.last_evaluated.get(snapshot_id),
            }
            for snapshot_id, rating in self.snapshot_ratings.items()
        }
        latest = snapshots[str(latest_id)] | {"snapshot_id": latest_id}
        self._write_json(
            self.checkpoint_dir / "trueskill_ratings.json",
            {
                "current":       latest,
                "latest":        latest,
                "snapshots":     snapshots,
                "match_batches": len(self.history),
            },
        )

    @staticmethod
    def _write_json(path: Path, payload: object) -> None:
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(
            json.dumps(payload, indent=2) + "\n", encoding="utf-8"
        )
        temporary.replace(path)


__all__ = ["TrueSkillEvaluator"]
