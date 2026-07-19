# JARL Staged Runtime

## Status

This document describes JARL's current runtime. The architecture is deliberately
explicit and does not infer execution order through a dependency graph.

```text
collect -> store -> sample -> prepare -> optimize -> maintain
```

Algorithms compose ordinary Python components in that order. Missing fields or
misordered stages fail where they are used rather than through a second schema
language.

## Principles

### Use RL terminology directly

Public records and batches use full names:

```text
observation
action
reward
next_obs
terminated
truncated
old_log_prob
policy_state
policy_version
baseline_value
baseline_next_value
value_version
advantage
returns
```

`next_obs` is retained because it remains concise beside `observation` and is a
widely understood transition name.

### Capture policy-time information with the action

The probability in PPO's denominator must describe the policy that selected the
action. `Policy.act()` therefore returns a `PolicyOutput` containing the action,
next recurrent state, and optional action log probability from the same
distribution evaluation.

Capture components determine which optional values enter storage:

```python
captures = (
    LogProbCapture(),
    PolicyVersionCapture(policy),
    ValueCapture(value_function),
)
```

Recurrent PPO adds `RecurrentStateCapture()`. Off-policy methods can omit all of
these captures.

Values are not an automatic collection responsibility. A recipe can use
`ValueCapture` or run `MaterializeValues` before updating the value function.
Recurrent value materialization remains intentionally unsupported until it has
an explicit state-unroll contract.

### Keep execution order visible

Transforms and learning stages run in configured list order. They do not declare
`requires` or `produces` sets, and the runtime does not topologically sort them.

```python
prepared = apply_transforms(
    rollout.steps,
    (SignRewards(), GAE()),
)
```

This trusts algorithm authors while keeping failures local and readable.

### Share mechanics, not semantics

`RolloutBuffer` and `ReplayBuffer` share tensor allocation, schema stability,
shape validation, and indexed writes through `TensorStorage`. Their lifetimes
remain distinct:

- A rollout is chronological, fills once, is consumed, and is cleared.
- Replay is circular, persistent, and sampled directly.

This avoids both duplicated storage mechanics and an ambiguous universal
`serve()` operation.

## Data

### PolicyOutput

```python
@dataclass
class PolicyOutput:
    action:     Tensor
    next_state: Tensor | None = None
    log_prob:   Tensor | None = None
```

`PolicyOutput` is the direct result of a policy evaluation, not a separate
collection abstraction.

### EnvStep

```python
@dataclass
class EnvStep:
    next_obs:    Tensor
    observation: Tensor
    reward:      Tensor
    terminated:  Tensor
    truncated:   Tensor
    info:        dict
```

`next_obs` is the actual transition successor. `observation` is the input for the
next policy evaluation after any automatic reset. They differ on an episode
boundary.

Keeping `terminated` and `truncated` separate permits return estimators to
bootstrap time-limit transitions while stopping trajectory recurrence.

### TensorBatch

`TensorBatch` is an immutable mapping of named tensors with common leading
dimensions. It supports indexing, selection, flattening, device transfer, and
explicit field addition or replacement.

Transforms return a new batch:

```python
return batch.with_fields(
    advantage=advantage,
    returns=advantage + value,
)
```

## Collection

`Runner` owns the current observation and optional recurrent policy state:

```python
runner = Runner(
    env,
    policy,
    rollout_buffer,
    captures=captures,
)
```

Each step performs:

1. Convert the observation to the policy device.
2. Evaluate `policy.act(observation, state)`.
3. Step the environment with the action.
4. Build the canonical transition.
5. Apply configured captures.
6. Append to the buffer.
7. Reset recurrent state where the episode ended.

CARL retains a specialized self-play loop because opponent selection, team
observation extraction, action combination, ratings, and match statistics are
not generic environment-runner concerns. It still uses the same `PolicyOutput`,
captures, records, rollout buffer, transforms, recurrent sampler, and PPO
optimizer.

## Storage

### TensorStorage

`TensorStorage` provides shared mechanics:

- lazy allocation from the first vector transition
- fixed field names after initialization
- stable shape and environment-count validation
- device conversion
- indexed tensor writes

### RolloutBuffer

Rollouts are stored as `[time, environment, ...]`. `finish()` returns an ordered
`Rollout(TensorBatch)` and supports a final partial rollout. `clear()` resets the
write position without reallocating tensors.

There is no wraparound, so GAE and recurrent sampling always receive temporal
order.

### ReplayBuffer

Replay stores vector environments as independent time lanes. Random transition
sampling gathers selected time/environment pairs directly. Window sampling:

- preserves time order within one environment lane
- maps logical chronology across physical ring wraparound
- rejects windows that cross an episode boundary
- transfers only selected tensors to the sample device

## Sampling

### Flat PPO

`RolloutMinibatches` flattens time and environment, shuffles each epoch, and
yields all samples including a partial final minibatch.

### Recurrent PPO

`RecurrentRolloutMinibatches` divides the rollout into contiguous sequences and
returns:

```python
@dataclass
class SequenceBatch:
    steps:         TensorBatch
    initial_state: Tensor
    reset:         Tensor
    valid:         Tensor
```

Only the first stored policy state initializes each sequence. The model unrolls
the remaining states and applies `reset` before observations following episode
boundaries.

### Replay windows

`ReplayBuffer.sample_windows(batch_size, length)` provides time-major windows
for n-step targets and recurrent off-policy methods.

## Preparation

Transforms are no-gradient batch calculations applied in explicit order.

Current transforms include:

- `SignRewards`
- `DiscriminatorReward`
- `MaterializeValues`
- `GAE`
- `DiscountedReturns`
- `NStepTarget`

GAE uses:

```text
bootstrap = not terminated
continues = not (terminated or truncated)
```

This bootstraps a truncation's final observation but does not carry advantage
recurrence into the reset episode.

## Optimization

`PPOOptimizer` owns minibatch epochs, policy/value evaluation, clipped losses,
gradient steps, scheduler advancement, metrics, and model-version increments.

`PPOLearner` validates that policy and value versions match the rollout, applies
configured transforms, then invokes the optimizer.

Version validation prevents an on-policy rollout from being reused after an
update or mixed across policy versions.

## Multi-stage Learning

Some algorithms require parameter updates between data transformations. GAIFO
is the primary example:

```text
train discriminator
    -> score rollout with updated discriminator
    -> materialize values if needed
    -> compute advantages and returns
    -> optimize PPO
```

`LearningProgram` executes coarse stages in list order. `LearningWorkspace`
holds named experience and TensorBoard metrics. It does not prevalidate stage
dependencies.

```python
program = LearningProgram((
    TrainDiscriminator(
        rollout="experience",
        expert_buffer=expert_buffer,
        discriminator=discriminator,
        optimizer_step=discriminator_step,
        batch_size=256,
    ),
    TransformRollout(
        rollout="experience",
        output="rewarded",
        transforms=(DiscriminatorReward(discriminator),),
    ),
    TransformRollout(
        rollout="rewarded",
        output="prepared",
        transforms=(GAE(reward_field="imitation_reward"),),
    ),
    OptimizePPO(
        rollout="prepared",
        optimizer=ppo_optimizer,
    ),
))
```

Expert transitions are sampled only by the discriminator stage. They are never
concatenated into the chronological policy rollout.

## Runtime

`Trainer` coordinates a runner, buffer, learner, and schedule. Clocks distinguish
vector steps, environment ticks, episodes, learner updates, and optimizer steps.

- `OnPolicySchedule` updates when a rollout is full and flushes a final partial
  rollout.
- `OffPolicySchedule` starts after a configured number of environment ticks and
  updates at a configured vector-step interval.

Target-network updates and replay-priority changes belong inside an off-policy
learner because their exact order affects the algorithm.

## Logging

TensorBoard is the sole metric backend. Console output is intentionally limited
to the current update count and global ticks.

Episode statistics are aggregated over the most recent 50 episodes before being
written to TensorBoard. Learner metrics retain their section names.

## Extension Guidelines

To add an algorithm:

1. Choose policy-time captures.
2. Choose rollout or replay storage.
3. Choose transition, window, flat, or recurrent sampling.
4. Write ordered preparation transforms.
5. Implement optimization order in a learner or coarse learning stages.
6. Select an on-policy or off-policy schedule.

Prefer ordinary Python control flow over additional schedulers or dependency
systems. Introduce a shared abstraction only after two concrete algorithms need
the same mechanism.
