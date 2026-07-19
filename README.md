# JARL (WIP)

JARL is under active development and supports implementing a wide variety of reinforcement learning algorithms.

JARL is written to be highly modular and allow for rapid prototyping of different RL algorithms.
Eventually, many existing algorithms will be implemented by default within JARL. Their core components will therefore be available to rearrange and refactor into more novel approaches.

## Installation

Install the project and its locked dependencies with [uv](https://docs.astral.sh/uv/):

```bash
uv sync
```

TensorBoard logging is available as an optional extra:

```bash
uv sync --extra logging
```

## Runtime Structure

JARL uses an explicit staged runtime:

```text
collect -> store -> sample -> prepare -> optimize -> maintain
```

- `Runner` performs environment interaction.
- Capture components choose optional policy-time fields such as action log
  probabilities, values, and recurrent state.
- `RolloutBuffer` owns ordered consumable on-policy data.
- `ReplayBuffer` owns persistent off-policy data and samples transitions or
  episode-safe windows directly.
- Ordered transforms derive rewards, advantages, returns, and targets.
- Flat and recurrent samplers choose minibatch temporal structure.
- Learners own optimizer ordering, target updates, and scheduler advancement.
- `LearningProgram` optionally sequences coarse stages such as discriminator
  training, learned-reward materialization, and PPO optimization.

The PPO collection boundary is explicit:

```python
captures = (
    LogProbCapture(),
    PolicyVersionCapture(policy),
    ValueCapture(critic),
)
rollout = RolloutBuffer(horizon=128, num_envs=8, device="cuda")
runner = Runner(env, policy, rollout, captures=captures)
```

`ppo.py` is a complete feed-forward example.

## Recurrent Networks

GRU and LSTM feature modules are built in and use time-major sequences:

```python
from jarl.modules import GRU, LSTM

gru = GRU(hidden_size=256, num_layers=2).build(observation_size)
state = gru.initial_state(batch_size)
features, state = gru(observations, state, reset=reset_mask)
```

GRU state is shaped `[batch, layers, hidden]`. LSTM packs hidden and cell state
into one tensor shaped `[batch, 2, layers, hidden]`, so both work with the same
collection and recurrent-sampling interfaces.
