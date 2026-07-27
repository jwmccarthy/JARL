import unittest
from types import SimpleNamespace
from unittest.mock import Mock

import torch

from jarl.data import TensorBatch
from jarl.learn import TransformRollout
from jarl.log.logger import Logger
from jarl.store import Rollout


class AddMetric:
    def __call__(self, batch, context):
        return batch.with_fields(metric=batch["value"] * 2)


class ReportingTests(unittest.TestCase):
    def test_registered_update_metric_is_added_and_formatted(self):
        logger = Logger()
        metrics = Mock()
        metrics.add_task.return_value = 7
        logger._metrics = metrics

        logger.register_progress_metric(
            "GAIfO", "imitation_reward", format_spec=",.4f"
        )
        logger.update({"GAIfO": {"imitation_reward": 1.23456}})

        metrics.add_task.assert_called_once_with(
            "imitation_reward", total=None, value="-"
        )
        metrics.update.assert_called_once_with(7, value="1.2346")

    def test_duplicate_progress_metric_is_rejected(self):
        logger = Logger()

        with self.assertRaisesRegex(ValueError, "already registered"):
            logger.register_progress_metric("Episode", "reward")

    def test_activity_replaces_the_secondary_progress_task(self):
        logger = Logger()
        progress = Mock()
        progress.tasks = {4: SimpleNamespace(total=10, completed=3)}
        logger._progress = progress
        logger._activity_task = 4

        logger.start_activity(10, "trueskill_matches")
        logger.advance_activity(3)
        logger.finish_activity()

        self.assertEqual(
            progress.update.call_args_list[0].kwargs,
            {"description": "trueskill_matches", "total": 10, "completed": 0},
        )
        progress.advance.assert_called_once_with(4, 3)
        self.assertEqual(progress.update.call_args_list[1].kwargs, {"completed": 10})

    def test_transform_rollout_reports_selected_learner_field(self):
        rollout = Rollout(
            TensorBatch(
                {
                    "value": torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
                    "learner_mask": torch.tensor(
                        [[True, False], [True, False]]
                    ),
                }
            )
        )
        stage = TransformRollout(
            AddMetric(),
            report_fields=("metric",),
            section="Custom",
        )

        transformed, metrics = stage.run(rollout)

        torch.testing.assert_close(
            transformed.steps["metric"],
            torch.tensor([[2.0, 4.0], [6.0, 8.0]]),
        )
        self.assertEqual(metrics, {"Custom": {"metric": 4.0}})


if __name__ == "__main__":
    unittest.main()
