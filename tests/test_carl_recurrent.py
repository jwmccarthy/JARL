import unittest

import numpy as np
import torch as th

from carl_ppo import CarlEnv, GRUBackbone, GRUPolicy, GRUValue


class FakeActionSpace:
    nvec = np.array([3, 3, 3, 2, 2, 3, 2])
    shape = (7,)


class FakeCarlEnv:
    obs_dim = 89
    n_cars = 2
    tick_skip = 8
    act_space = FakeActionSpace()


class CarlRecurrentTests(unittest.TestCase):
    def setUp(self):
        th.manual_seed(3)
        self.env = FakeCarlEnv()
        self.backbone = GRUBackbone(self.env.obs_dim, 16)
        self.policy = GRUPolicy(self.env, 16, self.backbone)
        self.critic = GRUValue(16, self.backbone)

    def test_immediate_action_log_probability_matches_evaluation(self):
        observation = th.randn(3, self.env.obs_dim)
        state = self.policy.initial_state(3)
        policy_output = self.policy.act(observation, state)
        evaluation = self.policy.evaluate_actions(
            observation.unsqueeze(0),
            policy_output.action.unsqueeze(0),
            state,
        )
        th.testing.assert_close(
            policy_output.log_prob,
            evaluation.log_prob.squeeze(0),
        )

    def test_native_packed_observations_expose_blue_raw_state(self):
        packed = th.arange(2 * 2 * 51).reshape(2, 2, 51)
        adapter = CarlEnv.__new__(CarlEnv)
        th.testing.assert_close(adapter.raw_state(packed), packed[:, 0])

    def test_reset_mask_restarts_policy_and_value_unrolls(self):
        observation = th.randn(4, 2, self.env.obs_dim)
        action = th.zeros(4, 2, 7, dtype=th.long)
        state = th.randn(2, 16)
        reset = th.tensor(
            [[False, False], [False, False], [True, True], [False, False]]
        )

        policy_full = self.policy.evaluate_actions(
            observation,
            action,
            state,
            reset=reset,
        ).log_prob
        policy_restart = self.policy.evaluate_actions(
            observation[2:],
            action[2:],
            th.zeros_like(state),
        ).log_prob
        th.testing.assert_close(policy_full[2:], policy_restart)

        value_full = self.critic.evaluate_values(observation, state, reset=reset)
        value_restart = self.critic.evaluate_values(
            observation[2:],
            th.zeros_like(state),
        )
        th.testing.assert_close(value_full[2:], value_restart)


if __name__ == "__main__":
    unittest.main()
