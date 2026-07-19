import unittest

import torch as th

from jarl.modules import GRU, LSTM


class RecurrentModuleTests(unittest.TestCase):
    def test_gru_single_step_and_sequence_shapes(self):
        gru = GRU(hidden_size=5, num_layers=2).build(3)

        step_output, step_state = gru(th.randn(4, 3))
        sequence_output, sequence_state = gru(th.randn(6, 4, 3))

        self.assertEqual(gru.feats, 5)
        self.assertEqual(step_output.shape, (4, 5))
        self.assertEqual(step_state.shape, (4, 2, 5))
        self.assertEqual(sequence_output.shape, (6, 4, 5))
        self.assertEqual(sequence_state.shape, (4, 2, 5))

    def test_lstm_uses_one_packed_tensor_state(self):
        lstm = LSTM(hidden_size=7, num_layers=2).build(3)

        output, state = lstm(th.randn(5, 4, 3))

        self.assertEqual(output.shape, (5, 4, 7))
        self.assertEqual(state.shape, (4, 2, 2, 7))

    def test_reset_restarts_before_corresponding_input(self):
        for recurrent_type in (GRU, LSTM):
            with self.subTest(recurrent_type=recurrent_type.__name__):
                recurrent = recurrent_type(hidden_size=5).build(3)
                value = th.randn(4, 2, 3)
                state = th.randn_like(recurrent.initial_state(2))
                reset = th.tensor(
                    [
                        [False, False],
                        [False, False],
                        [True, True],
                        [False, False],
                    ]
                )

                output, next_state = recurrent(value, state, reset)
                restarted_output, restarted_state = recurrent(
                    value[2:],
                    recurrent.initial_state(2),
                )

                th.testing.assert_close(output[2:], restarted_output)
                th.testing.assert_close(next_state, restarted_state)

    def test_single_step_reset_is_independent_per_batch_item(self):
        gru = GRU(hidden_size=4).build(3)
        value = th.randn(2, 3)
        state = th.randn(2, 1, 4)

        output, _ = gru(value, state, th.tensor([True, False]))
        reset_output, _ = gru(value[:1], gru.initial_state(1))
        retained_output, _ = gru(value[1:], state[1:])

        th.testing.assert_close(output[:1], reset_output)
        th.testing.assert_close(output[1:], retained_output)

    def test_initial_state_uses_requested_dtype_and_device(self):
        lstm = LSTM(hidden_size=4).build(3).double()

        state = lstm.initial_state(2)

        self.assertEqual(state.dtype, th.float64)
        self.assertEqual(state.device, next(lstm.parameters()).device)

    def test_recurrent_construction_options_are_forwarded(self):
        initialized = []

        def initialize(module):
            initialized.append(module)
            return module

        gru = GRU(
            hidden_size=4,
            num_layers=2,
            dropout=0.25,
            bias=False,
            init_func=initialize,
        ).build(3)

        self.assertIs(initialized[0], gru.rnn)
        self.assertEqual(gru.rnn.dropout, 0.25)
        self.assertFalse(gru.rnn.bias)

    def test_invalid_shapes_are_rejected(self):
        gru = GRU(hidden_size=4).build(3)
        value = th.randn(5, 2, 3)

        with self.assertRaisesRegex(ValueError, "recurrent input"):
            gru(th.randn(2, 3, 4, 5))
        with self.assertRaisesRegex(ValueError, "input has"):
            gru(th.randn(2, 4))
        with self.assertRaisesRegex(ValueError, "state has shape"):
            gru(value, th.zeros(2, 4))
        with self.assertRaisesRegex(ValueError, "reset has shape"):
            gru(value, reset=th.zeros(2, dtype=th.bool))

    def test_gradients_flow_through_reset_unroll(self):
        lstm = LSTM(hidden_size=4).build(3)
        value = th.randn(5, 2, 3, requires_grad=True)
        reset = th.tensor(
            [
                [False, False],
                [False, True],
                [False, False],
                [True, False],
                [False, False],
            ]
        )

        output, state = lstm(value, reset=reset)
        (output.sum() + state.sum()).backward()

        self.assertIsNotNone(value.grad)
        self.assertTrue(
            all(parameter.grad is not None for parameter in lstm.parameters())
        )


if __name__ == "__main__":
    unittest.main()
