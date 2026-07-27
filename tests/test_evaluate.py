import json
import tempfile
import unittest
from pathlib import Path

from jarl.collect.evaluate import TrueSkillEvaluator


class Pool:
    archive_ids = (0,)

    def __init__(self, checkpoint_dir):
        self.checkpoint_dir = checkpoint_dir

    @staticmethod
    def timesteps(snapshot_id):
        return snapshot_id


class FixedOpponentEvaluatorTests(unittest.TestCase):
    def test_fixed_anchor_participates_in_rating_history(self):
        with tempfile.TemporaryDirectory() as temporary:
            checkpoint_dir = Path(temporary)
            evaluator = TrueSkillEvaluator(
                policy=type("Policy", (), {"device": "cpu"})(),
                opponent_pool=Pool(checkpoint_dir),
                env_factory=None,
                logger=None,
                checkpoint_dir=checkpoint_dir,
                interval=1,
                num_matches=1,
                team_sizes=(1, 1),
                max_steps=1,
                opponents=1,
                draw_probability=0.1,
                seed=0,
                fixed_opponents={"Nexto": object()},
            )
            evaluator.history = [
                {"step": 1, "left": 0, "right": "anchor:Nexto", "outcomes": [1]}
            ]

            evaluator._recompute_ratings()
            evaluator._write_ratings(0)

            ratings = json.loads(
                (checkpoint_dir / "trueskill_ratings.json").read_text()
            )

        self.assertIn("anchor:Nexto", evaluator.snapshot_ratings)
        self.assertGreater(evaluator.snapshot_ratings[0].mu, 25.0)
        self.assertEqual(evaluator.rating_games["anchor:Nexto"], 1)
        self.assertIn("Nexto", ratings["anchors"])

    def test_evaluation_waits_for_its_interval(self):
        with tempfile.TemporaryDirectory() as temporary:
            checkpoint_dir = Path(temporary)
            evaluator = TrueSkillEvaluator(
                policy=type("Policy", (), {"device": "cpu"})(),
                opponent_pool=Pool(checkpoint_dir),
                env_factory=None,
                logger=None,
                checkpoint_dir=checkpoint_dir,
                interval=10,
                num_matches=1,
                team_sizes=(1, 1),
                max_steps=1,
                opponents=1,
                draw_probability=0.1,
                seed=0,
                fixed_opponents={"Nexto": object()},
            )

            self.assertFalse(evaluator.ready(9))
            self.assertTrue(evaluator.ready(10))


if __name__ == "__main__":
    unittest.main()
