# JARL

JARL provides PyTorch components for reinforcement learning.

## Requirements

JARL requires Python 3.11 or newer.

Install the package with:

```bash
uv sync
```

Install TensorBoard support with:

```bash
uv sync --extra logging
```

Install example dependencies with:

```bash
uv sync --extra examples
```

## Examples

Run recurrent PPO on LunarLander with:

```bash
uv run --extra examples python examples/ppo.py
```

Run GAIfO on LunarLander with:

```bash
uv run --extra examples python examples/gaifo.py
```

Both commands accept `--help`, `--total-timesteps`, `--num-envs`, `--rollout-steps`, `--device`, and `--checkpoint`.

## Environment Interface

An environment exposes `n_envs`. `reset()` returns a batched observation. `step()` returns observation, reward, terminated, truncated, and info.

JARL supports same step reset behavior. A terminal observation can be stored in `info["final_obs"]` or `info["final_observation"]`. The related mask can be stored in `info["_final_obs"]` or `info["_final_observation"]`.

When a truncated transition has no terminal observation, JARL does not bootstrap from the returned reset observation.

## Data And Reset Sampling

`TensorBatch` stores named tensors with a shared leading shape. `TensorDataset` stores a nonempty batch on one device.

`DatasetResetSampler` samples dataset rows for a reset mask. Its probability setting can leave part of the mask unchanged for the environment's normal reset behavior. It uses its own seeded generator and returns `None` when no override is selected. Optional reset transforms receive the sample and a `ResetContext`.

```python
dataset = TensorDataset(TensorBatch({"state": states}))
sampler = DatasetResetSampler(dataset, seed=0)
sample = sampler(reset_mask)
```

## Components

`Runner` collects transitions. Capture objects select extra policy data. `RolloutBuffer` stores ordered rollout data. `ReplayBuffer` stores reusable transitions and windows.

Transforms include GAE, discounted returns, n step targets, discriminator rewards, team rewards, and value materialization. Flat and recurrent samplers create update batches. `Algorithm` runs update stages in order. `Trainer` runs collection and update schedules.

Model components include encoders, MLP, CNN, GRU, LSTM, categorical policies, value functions, and `ActorCritic`. Recurrent `ActorCritic` uses shared actor and critic head and body objects.

## Value Scheduling

`ValueScheduler` applies arbitrary scalar schedules before collection and advances them using actual environment-step progress. Collection-time targets change continuously, while optimizer and loss targets take effect at the next update. Scheduled values are logged under `Schedule/*`.

```python
values = ValueScheduler(
    ScheduledValue.attribute(
        "gamma",
        gae,
        "gamma",
        LinearSchedule(0.99, 0.997),
    ),
    ScheduledValue(
        "learning_rate",
        LinearSchedule(1e-4, 5e-5),
        set_learning_rate,
    ),
)

trainer = Trainer(..., value_scheduler=values)
```

`ConstantSchedule`, `LinearSchedule`, and `MappedSchedule` are provided. Any callable from progress in `[0, 1]` to a scalar can be used as a schedule, and arbitrary objects can be updated through a setter callback.

## Self Play

`SnapshotPool` stores policy snapshots on CPU. `SelfPlayMatchmaker` assigns current and saved policies to vectorized matches. `SelfPlayRunner` excludes saved policy transitions from optimization through its learner mask.

A match is reassigned when an actor in that match finishes. `TeamSpirit` mixes individual reward with team mean reward. `TrueSkillEvaluator` plays both team assignments and writes ratings to `trueskill_ratings.json`.
