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
   `Algorithm` runs its update and transform stages in the order they are listed.

## PPO Example

The complete [LunarLander example](examples/ppo.py) builds the environment,
policy, value function, PPO update, and training loop from JARL components. Run
it with:

```bash
uv run --extra examples python examples/ppo.py
```

Use `--total-env-steps` for a shorter run or `--checkpoint PATH` to save the
trained policy.

## GAIfO Example

The [GAIfO example](examples/gaifo.py) collects expert LunarLander transitions
with Gymnasium's heuristic controller, trains a transition discriminator, and
uses its rewards for PPO:

```bash
uv run --extra examples python examples/gaifo.py
```
