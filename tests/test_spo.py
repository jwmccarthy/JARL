import unittest

import torch

from jarl.learn import SPOConfig, SPOLoss


class SPOLossTests(unittest.TestCase):
    def test_ratio_epsilon_must_be_positive(self):
        with self.assertRaisesRegex(ValueError, "ratio epsilon"):
            SPOLoss(None, None, SPOConfig(ratio_epsilon=0.0))

    def test_quadratic_surrogate_prefers_signed_epsilon_ratio(self):
        loss = SPOLoss(None, None, SPOConfig(ratio_epsilon=0.2))
        advantage = torch.tensor([1.0, -1.0])

        optimum = loss._policy_loss(advantage, torch.tensor([1.2, 0.8]))
        displaced = loss._policy_loss(advantage, torch.tensor([1.0, 1.0]))

        self.assertLess(optimum, displaced)


if __name__ == "__main__":
    unittest.main()
