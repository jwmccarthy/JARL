import tempfile
import unittest
from pathlib import Path

import torch

from jarl.collect import SelfPlayMatchmaker, SnapshotPool


class SnapshotPoolTests(unittest.TestCase):
    def test_close_finishes_background_checkpoint_write(self):
        with tempfile.TemporaryDirectory() as temporary:
            checkpoint_dir = Path(temporary) / "snapshots"
            pool = SnapshotPool(
                torch.nn.Linear(2, 1),
                max_size=3,
                snapshot_interval=1,
                checkpoint_dir=checkpoint_dir,
            )
            pool.close()

            checkpoint = checkpoint_dir / "policy_000000000000.pt"
            self.assertTrue(checkpoint.is_file())
            self.assertEqual(
                set(torch.load(checkpoint, weights_only=True)),
                {"weight", "bias"},
            )

    def test_recent_snapshots_form_active_window_but_initial_stays_archived(self):
        with tempfile.TemporaryDirectory() as temporary:
            pool = SnapshotPool(
                torch.nn.Linear(2, 1),
                max_size=3,
                snapshot_interval=1,
                checkpoint_dir=Path(temporary) / "snapshots",
            )
            for timestep in range(1, 5):
                pool.add(torch.nn.Linear(2, 1), timestep)
            pool.close()

            self.assertEqual(pool.ids, (2, 3, 4))
            self.assertEqual(pool.archive_ids, (0, 1, 2, 3, 4))
            self.assertEqual(pool.select_ids(2), (3, 4))

    def test_snapshot_sampling_favors_recent_policies(self):
        pool = SnapshotPool(
            torch.nn.Linear(2, 1),
            max_size=5,
            snapshot_interval=1,
            seed=7,
        )
        for timestep in range(1, 5):
            pool.add(torch.nn.Linear(2, 1), timestep)

        counts = {snapshot_id: 0 for snapshot_id in range(1, 5)}
        for _ in range(2000):
            counts[pool.sample_ids(1)[0]] += 1

        self.assertGreater(counts[4], counts[3])
        self.assertGreater(counts[3], counts[2])
        self.assertGreater(counts[2], counts[1])

    def test_matchmaking_weights_recent_active_snapshots(self):
        matchmaker = SelfPlayMatchmaker(
            num_matches=6000,
            team_sizes=(1, 1),
            current_fraction=0.0,
            historical_ids=(1, 2, 3),
            device="cpu",
            seed=11,
        )
        opponent_ids = matchmaker.opponent_ids[matchmaker.opponent_ids >= 0]
        counts = torch.bincount(opponent_ids, minlength=4)

        self.assertGreater(counts[3], counts[2])
        self.assertGreater(counts[2], counts[1])


if __name__ == "__main__":
    unittest.main()
