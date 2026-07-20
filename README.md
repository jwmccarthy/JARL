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

### Environment contract

JARL environments expose `n_envs`, return batched observations directly from
`reset()`, and return the Gymnasium five-tuple from `step()`. `Runner` converts
each tuple into an `EnvStep` for collection and learning.

`total_timesteps` and `global_t` count learner-controlled transitions. Standard
runners count every environment actor; self-play runners exclude historical
opponent actors.

Same-step autoreset is supported. Environments should provide terminal
observations as `info["final_obs"]` with an `info["_final_obs"]` mask. JARL
uses those observations for truncated-transition bootstrapping while advancing
the policy with the returned reset observations. If a truncated environment
does not expose its terminal observation, JARL conservatively disables
bootstrapping rather than using the next episode's reset state.

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

Use `--total-timesteps` to set the training budget or `--checkpoint PATH` to save the
trained policy.

## GAIfO Example

The [GAIfO example](examples/gaifo.py) collects expert LunarLander transitions
with Gymnasium's heuristic controller, trains a transition discriminator, and
uses its rewards for PPO:

```bash
uv run --extra examples python examples/gaifo.py
```

## Self-Play

JARL provides `SnapshotPool`, `SelfPlayMatchmaker`, and `SelfPlayRunner` for
two-team self-play. Matchmaking can divide vectorized matches between the
current policy and immutable historical snapshots. Current-v-current matches
train every actor; current-v-historical matches randomize the learner's team
and exclude historical-policy transitions from optimization.

Snapshots are retained on CPU in a bounded pool and selected opponents are
cached on the execution device. Match assignments persist for a complete
episode and are updated after same-step autoreset transitions have been stored.

`TeamSpirit` blends each individual reward with the mean reward of its team
before return estimation:

```text
reward = (1 - spirit) * individual + spirit * team_mean
```

A value of zero preserves individual rewards; one uses fully shared team
rewards.

`ActorCritic` composes existing actor and critic modules. Shared state is
enabled by passing the same head and body instances to both modules:

```python
head = FlattenEncoder()
body = GRU(hidden_size=256)

actor = MultiCategoricalPolicy(
    head=head,
    body=body,
    foot=MLP(dims=[]),
    action_codec=environment.action_codec,
)
critic = ValueFunction(
    head=head,
    body=body,
    foot=MLP(dims=[]),
)
model = ActorCritic(
    actor=actor,
    critic=critic,
    shared_state=True,
).build(environment)
```

The shared head/body executes once per request and each foot produces its own
output. Recurrence is a property of the selected body rather than the
actor-critic container. Use recurrent bodies with `RecurrentStateCapture` and
`RecurrentRolloutMinibatches`; self-play sequence batches retain historical
transitions for correct hidden-state unrolling while masking them out of PPO
losses.

The recurrent sampler can prune unused fields and separates reset-free
sequences for fused cuDNN execution. On-policy rollout buffers can expose
zero-copy views during updates.

An environment-provided joint action codec can be attached to the actor. JARL
masks invalid joint logits before sampling and applies the same mask when
reevaluating stored actions for PPO.
