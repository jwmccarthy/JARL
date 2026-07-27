import tempfile
import unittest
from pathlib import Path

import torch

from jarl.collect import SnapshotPool


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


if __name__ == "__main__":
    unittest.main()
