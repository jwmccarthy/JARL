import unittest

import torch

from jarl.transform.returns import discounted_suffix_sum


class DiscountedSuffixSumTests(unittest.TestCase):
    def test_matches_reverse_recurrence_across_episode_boundaries(self):
        value = torch.tensor(
            [[1.0, 10.0], [2.0, 20.0], [3.0, 30.0], [4.0, 40.0]]
        )
        continues = torch.tensor(
            [[True, True], [False, True], [True, False], [False, False]]
        )

        result = discounted_suffix_sum(value, continues, 0.5)

        torch.testing.assert_close(
            result,
            torch.tensor([[2.0, 27.5], [2.0, 35.0], [5.0, 30.0], [4.0, 40.0]]),
        )

    @unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
    def test_cuda_matches_cpu(self):
        value = torch.randn(32, 64)
        continues = torch.rand(32, 64) > 0.1

        expected = discounted_suffix_sum(value, continues, 0.99)
        actual = discounted_suffix_sum(value.cuda(), continues.cuda(), 0.99).cpu()

        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)


if __name__ == "__main__":
    unittest.main()
