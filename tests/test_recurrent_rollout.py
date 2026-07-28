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
            sequences_per_batch=2,
        )

        samples = list(sampler(batch))

        self.assertEqual(len(samples), 2)
        clean = next(sample for sample in samples if sample.reset is None)
        resetting = next(sample for sample in samples if sample.reset is not None)
        self.assertTrue(clean.steps["observation"].is_contiguous())
        self.assertTrue(resetting.steps["observation"].is_contiguous())
        self.assertFalse(resetting.reset[0].any())
        self.assertEqual(resetting.reset[1].sum().item(), 1)

    def test_tiny_reset_group_is_folded_into_balanced_batches(self):
        num_envs = 101
        terminated = torch.zeros(2, num_envs, dtype=torch.bool)
        terminated[0, 0] = True
        sequence_ids = torch.arange(num_envs).repeat(2, 1)
        batch = TensorBatch(
            {
                "sequence_id": sequence_ids,
                "policy_state": torch.zeros(2, num_envs, 1, 1),
                "critic_state": torch.zeros(2, num_envs, 1, 1),
                "terminated": terminated,
                "truncated": torch.zeros_like(terminated),
                "learner_mask": torch.ones_like(terminated),
            }
        )
        sampler = RecurrentRolloutMinibatches(
            sequence_length=2,
            sequences_per_batch=32,
            epochs=2,
        )

        samples = list(sampler(batch))
        self.assertEqual(
            sorted(sample.valid.shape[1] for sample in samples),
            [25, 25, 25, 25, 25, 25, 26, 26],
        )
        self.assertEqual(sum(sample.reset is not None for sample in samples), 2)

        for epoch in (samples[:4], samples[4:]):
            sampled_ids = torch.cat(
                [sample.steps["sequence_id"][0] for sample in epoch]
            )
            self.assertEqual(
                sampled_ids.sort().values.tolist(), list(range(num_envs))
            )
            resetting = next(sample for sample in epoch if sample.reset is not None)
            reset_id = (
                resetting.steps["sequence_id"][0] == 0
            ).nonzero(as_tuple=True)[0]
            self.assertEqual(reset_id.numel(), 1)
            self.assertTrue(resetting.reset[1, reset_id].item())


if __name__ == "__main__":
    unittest.main()
