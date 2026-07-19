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

Training is split into six stages:

1. **Collect**

   `Runner` uses the policy to step the environment. Capture components can
   also record values, action log probabilities, and recurrent state.

2. **Store**

   `RolloutBuffer` keeps ordered on-policy data until it is consumed.
   `ReplayBuffer` keeps off-policy data for reuse.

3. **Sample**

   Samplers turn stored data into flat or recurrent minibatches.

4. **Prepare**

   Transforms calculate the rewards, advantages, returns, and targets needed
   for training.

5. **Optimize**

   Learners run the optimizer steps that update each model.

6. **Maintain**

   Target networks and learning-rate schedules are updated after training.
   `LearningProgram` can coordinate multi-part updates when an algorithm needs
   them.

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
