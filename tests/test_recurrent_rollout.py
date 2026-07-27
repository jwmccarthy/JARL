import unittest

import torch

from jarl.data import TensorBatch
from jarl.sample import RecurrentRolloutMinibatches


class RecurrentRolloutMinibatchesTests(unittest.TestCase):
    def test_clean_sequences_skip_reset_masks_and_are_time_contiguous(self):
        terminated = torch.zeros(4, 2, dtype=torch.bool)
        terminated[0, 0] = True
        batch = TensorBatch(
            {
                "observation": torch.arange(32).reshape(4, 2, 4),
                "action": torch.zeros(4, 2, 1, dtype=torch.int64),
                "policy_state": torch.zeros(4, 2, 1, 3),
                "critic_state": torch.zeros(4, 2, 1, 3),
                "terminated": terminated,
                "truncated": torch.zeros_like(terminated),
                "learner_mask": torch.ones_like(terminated),
            }
        )
        sampler = RecurrentRolloutMinibatches(
            sequence_length=2,
            sequences_per_batch=4,
        )

        samples = list(sampler(batch))

        self.assertEqual(len(samples), 2)
        clean = next(sample for sample in samples if sample.reset is None)
        resetting = next(sample for sample in samples if sample.reset is not None)
        self.assertTrue(clean.steps["observation"].is_contiguous())
        self.assertTrue(resetting.steps["observation"].is_contiguous())
        torch.testing.assert_close(
            resetting.reset,
            torch.tensor([[False], [True]]),
        )


if __name__ == "__main__":
    unittest.main()
