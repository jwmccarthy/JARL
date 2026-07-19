# JARL Runtime Redesign

## Status

This document describes JARL's staged runtime. It is intentionally more specific
than a list of abstractions, but it is not an API stability promise.

The global training execution graph has been replaced with a small staged
runtime:

```text
collect -> store -> sample -> prepare -> optimize -> maintain
```

The stages are explicit and ordered. Components inside a stage declare the
fields they require and produce so configuration mistakes can be detected, but
the runtime does not construct a general-purpose dependency graph.

## Goals

- Share environment interaction, storage mechanics, batching, optimization,
  logging, and scheduling across on-policy and off-policy algorithms.
- Let an algorithm explicitly choose which policy-, value-, or model-dependent
  fields are captured during collection.
- Preserve the behavior-time context needed for correct PPO, action masking,
  recurrent policies, and stochastic preprocessing.
- Treat ordered rollouts and replay memory as different storage contracts.
- Make one-step, n-step, trajectory, and recurrent training natural.
- Make update order explicit enough for PPO epochs, delayed TD3 policy updates,
  target-network updates, and multiple critic steps.
- Keep custom GPU environments and self-play possible without forcing their
  logic into the generic runtime.
- Retain JARL's useful data-oriented style and small composable calculations.

## Non-goals

- A distributed actor/learner system in the first implementation.
- A universal declarative language for every RL algorithm.
- Automatic inference of arbitrary execution order.
- Hiding algorithm-specific control flow when writing it directly is clearer.
- Backward compatibility with every current class during the redesign.

## Why Not a Global Execution Graph?

The former `TrainGraph` was useful for ordering calculations such as values,
advantages, and returns. It was less suitable for expressing the full lifecycle
of an RL algorithm:

- Some data must be captured when the action is selected.
- On-policy rollouts are ordered and consumed, while replay is persistent and
  sampled directly.
- Recurrent batches need contiguous windows, initial state, and reset masks.
- Target updates occur once per learner step, not once per minibatch by
  accident.
- Algorithms often have simple but different control flow, such as three critic
  updates followed by one actor and target update.

A graph can encode these rules, but doing so introduces more concepts than the
algorithms need. Explicit stages make lifetime and ordering visible. Within the
`prepare` stage, an optional field resolver could later compute requested
derived fields lazily. That resolver should remain local and acyclic rather than
becoming the runtime itself.

## Design Principles

### Model-dependent data has provenance

Any stored statistic derived from a model is associated with the model version
and context needed to reproduce it. Examples include:

- action log probability
- value estimate
- recurrent state before the action
- action mask
- exploration noise state

Behavior log probability identifies the distribution that selected the action
and should normally be captured from that action decision. A value estimate is
less tightly coupled: it may be captured during collection or materialized
before the value function is updated. Lazy materialization requires the intended
model version and enough recurrent or preprocessing context to reproduce the
calculation.

### Collection fields are selected, not automatic

The runner records environment facts. An algorithm supplies capture components
for optional fields such as log probabilities, values, recurrent state, action
masks, or exploration metadata. Storage allocation follows that capture
specification.

### Storage owns lifetime and ordering

An on-policy rollout is drained after use. A replay buffer remains available
across updates. These should not be modes of one ambiguous `serve()` method.

### Sampling chooses temporal structure

Storage preserves experience. A sampler decides whether training receives
independent transitions, contiguous windows, complete episodes, or PPO
minibatches over a finalized rollout.

### Transforms are explicit and ordered

Transforms should be small, testable calculations such as GAE, n-step returns,
or observation normalization. Their configured order is their execution order.
Field declarations validate that order; they do not secretly reorder it.

### Learners own optimization control flow

A learner decides which losses run, how often they run, and when target modules
or schedulers advance. This avoids trying to encode algorithm control flow in a
generic update scheduler.

## Package Layout

One possible package layout is:

```text
jarl/
  data/
    batch.py            # Batch and field validation
    records.py          # ActionDecision and EnvStep
  collect/
    runner.py           # Generic environment interaction
    behavior.py         # Behavior protocol and common adapters
    capture.py          # Optional rollout/replay field capture
  store/
    rollout.py          # Ordered consumable rollouts
    replay.py           # Persistent transition replay
    sequence.py         # Episode-aware recurrent replay
  sample/
    rollout.py          # PPO flat/sequence minibatches
    replay.py           # Uniform/prioritized transition sampling
    window.py           # n-step and burn-in windows
  transform/
    base.py
    returns.py          # GAE, Monte Carlo, n-step, V-trace
    normalize.py
  learn/
    base.py
    program.py          # Ordered coarse-grained learning stages
    ppo.py
    sac.py
    td3.py
    steps.py            # Reusable optimizer-step helpers
  runtime/
    trainer.py
    clock.py
    trigger.py
```

This can be introduced gradually. Existing modules do not need to move before
their replacement is functional.

## Data Model

### Batch

`TensorBatch` keeps the useful central idea of the former `MultiTensor`: named
tensors with common leading dimensions. It replaces attribute mutation with a
small mapping API that makes field addition explicit.

```python
from collections.abc import Iterator, Mapping
from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass(frozen=True)
class Batch(Mapping[str, Tensor]):
    data: dict[str, Tensor]

    def __post_init__(self) -> None:
        validate_common_prefix(self.data)

    def __getitem__(self, key: str) -> Tensor:
        return self.data[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self.data)

    def __len__(self) -> int:
        return len(self.data)

    def select(self, *keys: str) -> "Batch":
        return Batch({key: self.data[key] for key in keys})

    def with_fields(self, **fields: Tensor) -> "Batch":
        overlap = self.data.keys() & fields.keys()
        if overlap:
            raise KeyError(f"fields already exist: {sorted(overlap)}")
        return Batch(self.data | fields)

    def replace_fields(self, **fields: Tensor) -> "Batch":
        missing = fields.keys() - self.data.keys()
        if missing:
            raise KeyError(f"fields do not exist: {sorted(missing)}")
        return Batch(self.data | fields)
```

The exact immutable wrapper is optional. The important rules are that transforms
cannot silently overwrite behavior data and that shape validation happens at
component boundaries.

Use flat, qualified field names initially rather than a deeply nested tensor
container:

```text
obs
act
rew
next_obs
terminated
truncated
behavior_log_prob
baseline_value
baseline_next_value
value_version
behavior_state
policy_version
adv
ret
td_target
valid
reset
```

These names work naturally with the current dictionary-based implementation.
Constants or a `Field` enum can prevent spelling mistakes later if needed.

### Action decision

The behavior returns its action and any ephemeral artifacts it calculated while
choosing that action:

```python
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ActionDecision:
    action: Tensor
    next_state: Tensor | None = None
    artifacts: dict[str, Any] = field(default_factory=dict)
```

An artifact is not automatically stored. For example, a stochastic policy will
usually expose `log_prob` because it already has the distribution, while a
shared actor-critic may expose reusable backbone features. The collection
capture plan decides which artifacts become experience and which additional
heads are evaluated. Artifacts may also include action masks, distribution
parameters, or exploration state.

### Environment step

Normalize environment output at one boundary:

```python
@dataclass
class EnvStep:
    next_obs: Tensor
    collector_obs: Tensor
    reward: Tensor
    terminated: Tensor
    truncated: Tensor
    info: object = None

    @property
    def done(self) -> Tensor:
        return self.terminated | self.truncated
```

Do not collapse `terminated` and `truncated` in storage. Most return estimators
stop recurrence at either boundary but permit value bootstrapping across a time
limit. `next_obs` is the actual successor used for transition bootstrapping;
`collector_obs` is the observation used for the next action after any automatic
reset. They differ at an episode boundary. Environment adapters are responsible
for preserving the final observation when the underlying environment auto-resets.
The exact return masks should be computed by the target transform.

## Behavior Interface

The collector should depend on a behavior interface, not directly on a policy
distribution class:

```python
from typing import Protocol


class Behavior(Protocol):
    @property
    def version(self) -> int: ...

    def initial_state(self, batch_size: int, device: torch.device): ...

    @torch.no_grad()
    def act(
        self,
        obs: Tensor,
        state: Tensor | None,
        *,
        deterministic: bool = False,
    ) -> ActionDecision: ...
```

Recurrent state is batch-first at this boundary: `[batch, *state_shape]`.
Adapters for PyTorch GRUs transpose to and from `[layers, batch, hidden]`
internally. This keeps state compatible with the leading dimensions of stored
experience.

The trainable model separately exposes learner-time evaluation:

```python
@dataclass
class Evaluation:
    log_prob: Tensor | None = None
    entropy: Tensor | None = None
    value: Tensor | None = None
    q: Tensor | tuple[Tensor, ...] | None = None
    extras: dict[str, Tensor] = field(default_factory=dict)


class TrainableAgent(Behavior, Protocol):
    def evaluate_actions(
        self,
        obs: Tensor,
        action: Tensor,
        state: Tensor | None = None,
    ) -> Evaluation: ...
```

Not every algorithm must use `TrainableAgent`. DQN can provide a behavior that
chooses epsilon-greedy actions and a learner that directly owns online and
target Q-networks. The runtime shares protocols without forcing all models into
one inheritance tree.

### Combined actor-critic computation

For PPO, `act()` can expose shared features without automatically evaluating the
value head:

```python
class PPOAgent(nn.Module):
    def act(self, obs, state, *, deterministic=False):
        features, next_state = self.backbone(obs, state)
        dist = self.policy_head.distribution(features, obs)
        action = dist.mode if deterministic else dist.sample()
        return ActionDecision(
            action=action,
            next_state=next_state,
            artifacts={
                "log_prob": dist.log_prob(action),
                "features": features,
            },
        )
```

This makes shared computation available without evaluating or persisting values.
An algorithm that does not need values simply omits the corresponding capture.

## Collection Captures

The runner always records environment facts: observations, actions, rewards,
termination flags, and policy version. Everything else is supplied by explicit
capture components.

```python
@dataclass
class CaptureContext:
    obs: Tensor
    state_in: Tensor | None
    decision: ActionDecision
    env_step: EnvStep
    behavior: Behavior


class Capture(Protocol):
    produces: frozenset[str]

    def output_specs(self, env_spec, behavior_spec) -> dict[str, FieldSpec]: ...

    @torch.no_grad()
    def __call__(self, context: CaptureContext) -> dict[str, Tensor]: ...
```

Common captures are small and reusable:

```python
class DecisionArtifact:
    def __init__(self, artifact: str, field: str):
        self.artifact = artifact
        self.field = field
        self.produces = frozenset({field})

    def __call__(self, context):
        return {self.field: context.decision.artifacts[self.artifact]}


class RecurrentStateCapture:
    produces = frozenset({"behavior_state"})

    def __call__(self, context):
        return {"behavior_state": context.state_in}


class ValueCapture:
    produces = frozenset({
        "baseline_value",
        "baseline_next_value",
        "value_version",
    })

    def __init__(self, estimator):
        self.estimator = estimator

    def __call__(self, context):
        features = context.decision.artifacts.get("features")
        if features is not None and hasattr(self.estimator, "value_from_features"):
            current = self.estimator.value_from_features(features)
        else:
            current = self.estimator.value(
                context.obs,
                context.state_in,
            )
        next_value = self.estimator.value(
            context.env_step.next_obs,
            context.decision.next_state,
        )
        return {
            "baseline_value": current,
            "baseline_next_value": next_value,
            "value_version": full_version_tensor(self.estimator.value_version),
        }
```

The PPO collection specification is then explicit:

```python
ppo_captures = [
    DecisionArtifact("log_prob", "behavior_log_prob"),
    ValueCapture(agent),
]
```

Recurrent PPO additionally uses `RecurrentStateCapture()`. SAC might use no
captures at all. An imitation method may capture observations and actions only,
then derive learned rewards later.

The capture plan belongs to the collection recipe, not to `RolloutBuffer` or
`ReplayBuffer`. Either sink can therefore receive optional values, masks,
embeddings, or model metadata when a particular algorithm needs them.

Captures are not arbitrary lifecycle hooks. They can inspect one completed
interaction and add declared tensor fields, but they do not update parameters,
sample storage, or trigger learning. Construction validates duplicate output
fields and uses `output_specs()` to allocate storage. The abbreviated capture
implementations above omit straightforward spec methods for readability.

## Collection

### Generic runner

The runner owns interaction state but does not know PPO, SAC, or GAE:

```python
class Runner:
    def __init__(self, env, behavior: Behavior, sink, captures=(), callbacks=()):
        self.env = env
        self.behavior = behavior
        self.sink = sink
        self.captures = captures
        self.callbacks = callbacks
        self.obs = env.reset()
        self.state = behavior.initial_state(env.num_envs, self.obs.device)

    @torch.no_grad()
    def step(self) -> EnvStep:
        state_in = self.state
        decision = self.behavior.act(self.obs, state_in)
        env_step = self.env.step(decision.action)

        record = {
            "obs": self.obs,
            "act": decision.action,
            "rew": env_step.reward,
            "next_obs": env_step.next_obs,
            "terminated": env_step.terminated,
            "truncated": env_step.truncated,
            "policy_version": full_version_tensor(self.behavior.version),
        }
        context = CaptureContext(
            obs=self.obs,
            state_in=state_in,
            decision=decision,
            env_step=env_step,
            behavior=self.behavior,
        )
        for capture in self.captures:
            fields = capture(context)
            reject_duplicate_fields(record, fields)
            record.update(fields)
        self.sink.append(record)

        self.obs = env_step.collector_obs
        self.state = reset_state(decision.next_state, env_step.done)
        for callback in self.callbacks:
            callback.after_step(env_step)
        return env_step
```

The production implementation should avoid building Python dictionaries on a
hot GPU path if profiling shows that it matters. The interface remains the
same if storage writes into preallocated tensors directly.

The simple `ValueCapture` evaluates `baseline_next_value` after every step,
matching the current CARL collector. An optimized implementation can avoid most
of these extra evaluations by filling an ordinary transition's next value from
the following action decision. It must still evaluate the true final observation
at a truncation and the rollout's final successor. Both implementations
materialize the same stored field before learning begins.

### Specialized interaction

CARL self-play has team observation extraction, opponent selection, multiple
hidden states, action recombination, rating updates, and match statistics. The
generic runner should not absorb all of that.

Two clean options are:

1. Implement a `CarlBehavior` that presents one combined behavior interface and
   keeps opponent state internally.
2. Keep a specialized `CarlRunner` that writes the same record schema into a
   standard `RolloutBuffer`.

The second option is preferable if the interaction semantics remain highly
specialized. Shared data, storage, target, sampling, and learner interfaces are
more valuable than forcing every environment through one runner.

### Policy versions

Increment `behavior.version` after each successful learner update that changes
the behavior parameters. `RolloutBuffer` should reject mixed versions by
default. Replay permits mixed versions.

This gives on-policy validation a concrete invariant:

```python
if rollout["policy_version"].unique().numel() != 1:
    raise RuntimeError("on-policy rollout contains multiple policy versions")
```

## Storage

### Separate contracts

Do not retain one `Buffer.serve()` abstraction. Use separate types even if they
share allocation utilities.

```python
class ExperienceSink(Protocol):
    def append(self, record: dict[str, Tensor]) -> None: ...


class RolloutBuffer(ExperienceSink):
    @property
    def full(self) -> bool: ...
    def finish(self) -> "Rollout": ...
    def clear(self) -> None: ...


class ReplayBuffer(ExperienceSink):
    def ready(self, sample_size: int) -> bool: ...
    def sample(self, batch_size: int) -> TensorBatch: ...
    def sample_windows(self, batch_size: int, length: int) -> TensorBatch: ...
```

### Rollout buffer

`RolloutBuffer` is fixed-size, chronological, and consumable:

```python
@dataclass(frozen=True)
class Rollout:
    steps: TensorBatch

    def with_steps(self, steps: TensorBatch) -> "Rollout":
        return Rollout(steps=steps)


class TensorRolloutBuffer:
    def __init__(self, horizon, num_envs, specs, device):
        self.horizon = horizon
        self.num_envs = num_envs
        self.storage = allocate(specs, (horizon, num_envs), device)
        self.position = 0

    def append(self, record):
        if self.position == self.horizon:
            raise RuntimeError("rollout is full")
        validate_record(record, self.storage)
        assign_at(self.storage, self.position, record)
        self.position += 1

    def finish(self):
        if self.position == 0:
            raise RuntimeError("rollout is empty")
        return Rollout(TensorBatch({
            key: value[:self.position]
            for key, value in self.storage.items()
        }))
```

There is no wraparound. Clearing changes `position` to zero after training.
If values are captured, the final transition already contains
`baseline_next_value`, and `value_version` identifies its value function.
`policy_version` similarly identifies the behavior distribution. PPO validates
that each version field is uniform and still matches the trainable model before
optimization.

### Replay buffer

Replay is a circular buffer, but it samples selected indices directly rather
than converting or returning the entire allocation:

```python
class TensorReplayBuffer:
    def sample(self, spec):
        starts = self.index_sampler.sample(
            population=len(self),
            batch_size=spec.batch_size,
        )
        return self.window_reader.read(
            starts,
            length=spec.burn_in + spec.steps + spec.bootstrap_steps,
        )
```

Replay needs episode identifiers or boundary flags so windows never cross an
episode incorrectly. A generation counter prevents stale priority updates after
ring slots have been overwritten.

### CPU and GPU storage

Allocation and transfer policy should be configuration, not separate algorithm
code:

```python
ReplayConfig(
    capacity=1_000_000,
    storage_device="cpu",
    sample_device="cuda",
    pin_memory=True,
)
```

CARL can use CUDA rollout storage; Atari replay can use pinned CPU memory.

## Sampling

Sampling is separated from storage because the same data can be viewed in
different temporal shapes.

### Flat rollout minibatches

```python
class RolloutMinibatches:
    def __init__(self, batch_size: int, epochs: int): ...

    def __call__(self, rollout: Batch):
        flat = flatten_time_env(rollout)
        for _ in range(self.epochs):
            for indices in shuffled_batches(len(flat), self.batch_size):
                yield flat[indices]
```

### Recurrent rollout minibatches

Recurrent samples expose only one initial state per sequence:

```python
@dataclass
class SequenceBatch:
    steps: Batch                  # [time, batch, ...]
    initial_state: Tensor         # [batch, *state_shape]
    valid: Tensor                 # [time, batch]
    reset: Tensor                 # [time, batch]
```

The sampler obtains `initial_state` from `behavior_state` at each selected
sequence start. It does not pass the full stored state sequence into the model.
The model unrolls from the initial state, applying `reset` at episode boundaries.

For replay-based recurrent methods, add `burn_in`:

```text
| burn-in, no gradient | train window | bootstrap steps |
```

Burn-in reconstructs current-network hidden state before the train window.
Stored behavior state remains useful for collection provenance but should not be
assumed to equal current-network state after many off-policy updates.

### N-step windows

A replay sample specification should state temporal requirements explicitly:

```python
ReplaySampleSpec(
    batch_size=256,
    steps=5,
    burn_in=0,
    bootstrap_steps=1,
)
```

The resulting tensors remain time-major until a target transform reduces them.

## Transforms and Validation

### Ordered transform protocol

```python
class Transform(Protocol):
    requires: frozenset[str]
    produces: frozenset[str]

    def __call__(self, batch: Batch, context: "PrepareContext") -> Batch: ...
```

Preparation is deliberately simple:

```python
def apply_transforms(batch, transforms, context):
    available = set(batch)
    for transform in transforms:
        missing = transform.requires - available
        if missing:
            raise ValueError(
                f"{type(transform).__name__} requires missing fields {missing}"
            )
        duplicate = transform.produces & available
        if duplicate:
            raise ValueError(
                f"{type(transform).__name__} would overwrite {duplicate}"
            )
        batch = transform(batch, context)
        available.update(transform.produces)
    return batch
```

Validate the pipeline once during construction and optionally assert actual
outputs in development mode. Do not topologically sort. If one transform must
follow another, the recipe should show that order.

### GAE

GAE consumes baseline values for both sides of each transition. Those values may
have been captured or materialized:

```python
class GAE:
    produces = frozenset({"adv", "ret"})

    def __init__(self, gamma=0.99, lambda_=0.95, reward_field="rew"):
        self.gamma = gamma
        self.lambda_ = lambda_
        self.reward_field = reward_field
        self.requires = frozenset({
            reward_field,
            "terminated",
            "truncated",
            "baseline_value",
            "baseline_next_value",
        })

    @torch.no_grad()
    def __call__(self, batch, context):
        values = batch["baseline_value"]
        next_values = batch["baseline_next_value"]
        bootstrap_mask = ~batch["terminated"]
        continue_mask = ~(batch["terminated"] | batch["truncated"])
        delta = (
            batch[self.reward_field]
            + self.gamma * next_values * bootstrap_mask
            - values
        )

        adv = torch.zeros_like(delta)
        carry = torch.zeros_like(delta[-1])
        for t in reversed(range(len(delta))):
            carry = delta[t] + (
                self.gamma * self.lambda_ * continue_mask[t] * carry
            )
            adv[t] = carry
        return batch.with_fields(adv=adv, ret=adv + values)
```

Environment semantics sometimes require different truncation handling. Keep the
masks visible and test them rather than encoding them in a generic `done` field.
Algorithms that do not use a value baseline can choose a Monte Carlo or
discounted-return transform and derive advantages without adding any value
fields.

### Value materialization

`MaterializeValues` is the preparation-stage alternative to `ValueCapture`. It
runs before any stage that updates the value function:

```python
class MaterializeValues:
    requires = frozenset({"obs", "next_obs"})
    produces = frozenset({
        "baseline_value",
        "baseline_next_value",
        "value_version",
    })

    def __init__(self, estimator):
        self.estimator = estimator

    @torch.no_grad()
    def __call__(self, rollout, context):
        if self.estimator.value_version != context.expected_value_version:
            raise RuntimeError("value function changed before materialization")

        if "behavior_state" in rollout:
            values, next_values = evaluate_recurrent_values(
                self.estimator,
                rollout,
            )
        else:
            values = self.estimator.value(rollout["obs"], None)
            next_values = self.estimator.value(rollout["next_obs"], None)

        return rollout.with_fields(
            baseline_value=values,
            baseline_next_value=next_values,
            value_version=full_version_tensor(self.estimator.value_version),
        )
```

For recurrent data, `evaluate_recurrent_values` uses contiguous sequences,
stored initial states, and reset masks. It must evaluate true final observations
at truncations rather than auto-reset observations. A shared actor/value
backbone may make collection-time capture cheaper than this second rollout
unroll; the recipe chooses that trade-off.

The value version may be the same counter as the policy version for a shared
actor-critic, or a separate counter when actor and critic updates are scheduled
independently.

### N-step target

```python
class NStepQTarget:
    requires = frozenset({"rew", "terminated", "next_obs"})
    produces = frozenset({"td_target"})

    @torch.no_grad()
    def __call__(self, window, context):
        reward_sum, discount, bootstrap_mask = discounted_prefix(
            window["rew"], window["terminated"], self.gamma
        )
        target_q = context.target_agent.value(window["next_obs"][-1])
        target = reward_sum + discount * bootstrap_mask * target_q
        return reduce_window(window).with_fields(td_target=target)
```

SAC, DQN, TD3, and distributional variants can supply different bootstrap
evaluators while sharing window extraction and discounted-prefix utilities.

### Lazy behavior materialization

Post-rollout behavior computation should be an explicit exceptional component:

```python
class BehaviorMaterializer:
    def __init__(self, snapshot_store):
        self.snapshot_store = snapshot_store

    def log_prob(self, batch):
        version = require_single_version(batch["policy_version"])
        policy = self.snapshot_store.get(version)
        require_fields(batch, policy.reproduction_fields)
        return policy.evaluate_actions(
            batch["obs"],
            batch["act"],
            batch.get("behavior_state"),
        ).log_prob
```

This should not be the default PPO path. It is useful when a feed-forward actor
can avoid expensive behavior statistics on latency-sensitive environment
workers and immutable snapshots are already available.

## Learning

### Ordered learning programs

Some algorithms have several semantically distinct learning stages. GAIFO is a
representative example:

```text
fit discriminator on agent and expert transitions
    -> score the rollout with the updated discriminator
    -> materialize values if they were not captured
    -> compute advantages and returns
    -> optimize the actor-critic
```

These are coarser operations than batch transforms. A transform is a pure or
no-gradient batch-to-batch calculation. A learning stage may sample another
source, update parameters, run several minibatch epochs, and publish a new batch
artifact.

Use an explicitly ordered program rather than extending the field graph:

```python
@dataclass
class LearningWorkspace:
    artifacts: dict[str, object]
    metrics: MeanMetrics

    def require(self, name: str):
        try:
            return self.artifacts[name]
        except KeyError:
            raise RuntimeError(f"missing learning artifact {name!r}") from None

    def publish(self, name: str, value: object):
        if name in self.artifacts:
            raise RuntimeError(f"learning artifact {name!r} already exists")
        self.artifacts[name] = value


class LearningStage(Protocol):
    requires: frozenset[str]
    produces: frozenset[str]

    def run(self, workspace: LearningWorkspace) -> None: ...


class LearningProgram:
    def __init__(self, stages):
        self.stages = stages
        validate_stage_order(stages, initially_available={"source"})

    def update(self, source):
        workspace = LearningWorkspace(
            artifacts={"source": source},
            metrics=MeanMetrics(),
        )
        for stage in self.stages:
            stage.run(workspace)
        return workspace.metrics.compute()
```

Validation follows configured order and only checks named data artifacts. It
does not reorder stages. Parameter mutations are visible in the stage sequence
and should be documented through names such as `TrainDiscriminator` and
`OptimizePPO`, not disguised as produced tensor fields.

For algorithms with unique control flow, a handwritten learner using the same
components is equally valid and often preferable. `LearningProgram` earns its
place only if several algorithms benefit from configuring coarse stages.

### GAIFO example

A modular GAIFO program could be assembled as:

```python
gaifo = LearningProgram([
    DrainRollout(output="rollout"),
    TrainDiscriminator(
        agent_data="rollout",
        expert_source=expert_replay,
        discriminator=discriminator,
        epochs=discriminator_epochs,
    ),
    TransformArtifact(
        source="rollout",
        output="rewarded_rollout",
        transforms=[
            DiscriminatorReward(
                discriminator,
                output_field="imitation_rew",
            ),
        ],
    ),
    TransformArtifact(
        source="rewarded_rollout",
        output="valued_rollout",
        transforms=[
            MaterializeValues(estimator=agent),
        ],
        skip_if_fields_exist={"baseline_value", "baseline_next_value"},
    ),
    TransformArtifact(
        source="valued_rollout",
        output="prepared_rollout",
        transforms=[
            GAE(
                reward_field="imitation_rew",
                gamma=0.99,
                lambda_=0.95,
            ),
        ],
    ),
    OptimizePPO(
        source="prepared_rollout",
        agent=agent,
        minibatches=RolloutMinibatches(batch_size=256, epochs=4),
    ),
])
```

`TrainDiscriminator` samples agent transitions from the named rollout artifact
and expert transitions from its configured source. It completes all requested
discriminator optimizer steps before the next stage begins.

Its implementation can remain ordinary, local code:

```python
class TrainDiscriminator:
    requires = frozenset({"rollout"})
    produces = frozenset()

    def run(self, workspace):
        rollout = workspace.require("rollout")
        agent_data = flatten_time_env(rollout.steps).select("obs", "next_obs")

        for _ in range(self.epochs):
            for agent_batch in self.agent_sampler(agent_data):
                expert_batch = self.expert_source.sample(len(agent_batch))
                logits = torch.cat([
                    self.discriminator(agent_batch),
                    self.discriminator(expert_batch),
                ])
                labels = torch.cat([
                    torch.ones(len(agent_batch), device=logits.device),
                    torch.zeros(len(expert_batch), device=logits.device),
                ])
                loss = binary_cross_entropy_with_logits(logits, labels)
                self.optimizer_step(loss)
                workspace.metrics.add("discriminator", "loss", loss.item())

        self.discriminator.increment_version()


class TransformArtifact:
    def run(self, workspace):
        source = workspace.require(self.source)
        steps = apply_transforms(
            source.steps,
            self.transforms,
            self.context(source),
        )
        output = source.with_steps(steps)
        workspace.publish(self.output, output)
```

The exact discriminator label convention is configurable, but logits are
preferable to probabilities for numerical stability.

`DiscriminatorReward` is a no-gradient transform. It does not overwrite the
environment reward; it adds `imitation_rew`, making reward provenance visible.
The updated discriminator is used because stage order is explicit. A reward
model version can also be stored if exact reproducibility matters.

```python
class DiscriminatorReward:
    requires = frozenset({"obs", "next_obs"})
    produces = frozenset({"imitation_rew", "reward_model_version"})

    @torch.no_grad()
    def __call__(self, batch, context):
        logits = self.discriminator(batch["obs"], batch["next_obs"])
        # The discriminator convention above labels policy transitions as one.
        reward = torch.nn.functional.softplus(-logits)
        return batch.with_fields(
            imitation_rew=reward,
            reward_model_version=full_version_tensor(
                self.discriminator.version
            ),
        )
```

This avoids concatenating expert transitions into the policy rollout itself.
Expert and policy batches are joined only inside `TrainDiscriminator`; reward
materialization then operates on the original, chronologically ordered rollout.
That separation is especially important for recurrent sampling and GAE.

`MaterializeValues` illustrates why values should not be automatic collector
fields. PPO can use `ValueCapture` when recurrent context or throughput makes it
preferable, while GAIFO may compute values after discriminator fitting but
before any actor-critic update. The materializer must have the recurrent initial
states and reset masks needed to reproduce the value sequence. It publishes a
new artifact rather than mutating the original rollout.

The policy log probability is different: PPO's denominator identifies the
behavior distribution that selected the action, so it should normally be
captured from the action decision. Recomputing it later is safe only through the
explicit immutable-snapshot mechanism described earlier.

The same stage vocabulary supports other multi-stage methods:

- GAIL or AIRL changes the discriminator inputs and reward transform.
- Model-based RL alternates model fitting, synthetic rollout generation, and
  policy updates.
- Offline-to-online methods can run representation or critic pretraining before
  enabling environment collection.
- Population methods can evaluate, select, and then optimize policies as
  separate coarse stages.

### Handwritten GAIFO learner

If the stage wrapper adds little value, preserve exactly the same boundaries in
ordinary Python:

```python
class GAIFOLearner:
    def update(self, rollout):
        discriminator_metrics = self.discriminator_trainer.update(
            agent_data=rollout,
            expert_source=self.expert_replay,
        )
        rewarded = self.reward_transform(rollout)
        valued = self.value_materializer(rewarded) if self.needs_values else rewarded
        prepared = self.gae(valued)
        ppo_metrics = self.ppo_optimizer.update(prepared)
        return merge_metrics(discriminator_metrics, ppo_metrics)
```

The components and data contracts provide modularity; the generic program is a
convenience, not a requirement.

### Reusable optimizer step

Keep loss calculation separate from parameter stepping, but avoid making every
loss an independently scheduled global update:

```python
class OptimizerStep:
    def __init__(self, modules, optimizer, max_grad_norm=None): ...

    def __call__(self, loss: Tensor) -> None:
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if self.max_grad_norm is not None:
            clip_grad_norm_(unique_parameters(self.modules), self.max_grad_norm)
        self.optimizer.step()
```

Shared parameter handling must be explicit. PPO with a shared backbone normally
uses one combined loss and one optimizer step. Separate actor and critic
optimizers must either have disjoint parameters or define intentional shared
parameter semantics.

### Learner protocol

```python
class Learner(Protocol):
    def update(self, source) -> dict[str, float]: ...
```

The source may be a finalized rollout or replay buffer. This broad protocol is
intentional: forcing PPO epochs and SAC replay updates through exactly the same
inner iterator makes control flow less clear, not more reusable.

Reusable pieces live below the learner:

- samplers
- transforms
- loss functions
- optimizer steps
- target-network utilities
- metric accumulation

### PPO optimizer and learner

The standard sketch below shows a combined actor/value loss for shared-parameter
PPO. In the implementation, the terms can be configured as a fixed list of
objectives evaluated on the same minibatch:

```python
objectives = [
    ClippedPolicyObjective(),
    EntropyObjective(coef=0.01),
    ClippedValueObjective(coef=0.5),  # Optional.
]
```

Each objective declares required batch fields and evaluation outputs. Omitting
the value objective means no value targets are required by optimization, though
the chosen advantage transform may still require a baseline. This is simple
loss composition, not a scheduled execution graph.

```python
class PPOOptimizer:
    def __init__(self, agent, minibatches, optimizer_step, config):
        self.agent = agent
        self.minibatches = minibatches
        self.optimizer_step = optimizer_step
        self.config = config

    def update(self, data):
        metrics = MeanMetrics()

        for batch in self.minibatches(data):
            evaluation = self.agent.evaluate_actions(
                batch["obs"],
                batch["act"],
                batch.get("initial_state"),
            )
            log_ratio = evaluation.log_prob - batch["behavior_log_prob"]
            ratio = log_ratio.exp()
            advantage = normalize(batch["adv"])
            policy_loss = -torch.minimum(
                advantage * ratio,
                advantage * ratio.clamp(
                    1 - self.config.clip,
                    1 + self.config.clip,
                ),
            ).mean()
            value_loss = clipped_value_loss(
                evaluation.value,
                batch["baseline_value"],
                batch["ret"],
                self.config.value_clip,
            )
            entropy_loss = -evaluation.entropy.mean()
            loss = (
                policy_loss
                + self.config.value_coef * value_loss
                + self.config.entropy_coef * entropy_loss
            )
            self.optimizer_step(loss)
            metrics.add(...)

        self.agent.increment_version()
        return metrics.compute()


class PPOLearner:
    def __init__(self, transforms, optimizer):
        self.transforms = transforms
        self.optimizer = optimizer

    def update(self, rollout):
        data = apply_transforms(
            rollout.steps,
            self.transforms,
            PrepareContext(rollout=rollout),
        )
        return self.optimizer.update(data)
```

The recurrent variant uses a sequence minibatch implementation and the same
optimizer. `evaluate_actions` receives `initial_state`, and losses use `valid`
to exclude padded timesteps. GAIFO reuses `PPOOptimizer` directly after its own
reward and target preparation stages.

### SAC learner

SAC makes update ordering explicit in ordinary Python:

```python
class SACLearner:
    def update(self, replay):
        metrics = MeanMetrics()
        for _ in range(self.config.gradient_steps):
            window = replay.sample(self.config.sample_spec)
            batch = apply_transforms(window, self.transforms, self.context)

            critic_loss, td_error = self.critic_loss(batch)
            self.critic_step(critic_loss)

            policy_loss = self.policy_loss(batch["obs"])
            self.policy_step(policy_loss)

            if self.learn_temperature:
                temperature_loss = self.temperature_loss(batch["obs"])
                self.temperature_step(temperature_loss)

            polyak_update(self.target_critic, self.critic, self.config.tau)
            replay.update_priorities(batch["index"], td_error.detach().abs())
            metrics.add(...)

        self.agent.increment_version()
        return metrics.compute()
```

TD3 can use a normal conditional for delayed actor updates. This is easier to
read and test than encoding update ratios in a global graph.

## Runtime and Scheduling

### Explicit clocks

Track units separately:

```python
@dataclass
class Clock:
    vector_steps: int = 0
    env_steps: int = 0
    episodes: int = 0
    learner_updates: int = 0
    optimizer_steps: int = 0
```

Configuration names should include units, such as `rollout_vector_steps`,
`learning_starts_env_steps`, and `target_update_optimizer_steps`.

### Trainer

A small trainer coordinates collection and learning:

```python
class Trainer:
    def __init__(self, runner, source, learner, schedule, callbacks=()): ...

    def run(self, total_env_steps):
        while self.clock.env_steps < total_env_steps:
            env_step = self.runner.step()
            self.clock.record_env_step(env_step)

            if self.schedule.should_learn(self.clock, self.source):
                metrics = self.learner.update(self.source)
                self.clock.record_update(metrics)
                self.schedule.after_update(self.clock)

            for callback in self.callbacks:
                callback.after_iteration(self.clock)
```

For on-policy training, `source` is a rollout buffer and `should_learn` means the
rollout is full. The runtime finalizes it, validates captured model-version
fields, runs the configured learning stages, and clears it.

For off-policy training, `source` is replay and the schedule checks learning
starts and an update-to-data ratio. The learner samples replay directly.

The trainer may have separate `OnPolicySchedule` and `OffPolicySchedule`
implementations. Sharing a tiny protocol is better than one configuration class
with many irrelevant options.

### Maintenance and callbacks

Callbacks are appropriate for effects that do not alter optimization semantics:

- logging
- checkpointing
- evaluation
- episode statistics
- progress display

Target-network updates, optimizer schedulers, and replay priorities belong in
the learner because their ordering affects the algorithm.

Self-play opponent selection can be a runner callback only if it does not need
to change action construction. Otherwise it belongs in the specialized runner
or behavior.

## Example Assembly

### Feed-forward PPO

```python
agent = PPOAgent(...)
captures = [
    DecisionArtifact("log_prob", "behavior_log_prob"),
    ValueCapture(agent),
]
rollout = TensorRolloutBuffer(
    horizon=128,
    num_envs=8,
    specs=rollout_specs(env, captures),
    device="cuda",
)
runner = Runner(env, agent, rollout, captures=captures)
learner = PPOLearner(
    transforms=[GAE(gamma=0.99, lambda_=0.95)],
    optimizer=PPOOptimizer(
        agent=agent,
        minibatches=RolloutMinibatches(batch_size=256, epochs=4),
        optimizer_step=OptimizerStep(
            agent,
            torch.optim.Adam(agent.parameters(), lr=2.5e-4),
            max_grad_norm=0.5,
        ),
        config=PPOConfig(clip=0.1),
    ),
)
trainer = Trainer(
    runner,
    rollout,
    learner,
    schedule=OnPolicySchedule(horizon=128),
)
trainer.run(total_env_steps=10_000_000)
```

No automatic `ComputeValues` or `ComputeLogProbs` transform is present. This PPO
recipe explicitly captures both fields. Another recipe can omit value capture
and run `MaterializeValues` before GAE instead.

### Recurrent PPO

Only the agent, rollout specification, and sampler change:

```python
agent = RecurrentPPOAgent(...)
captures = [
    DecisionArtifact("log_prob", "behavior_log_prob"),
    ValueCapture(agent),
    RecurrentStateCapture(),
]
rollout = TensorRolloutBuffer(..., specs=rollout_specs(env, captures))
runner = Runner(env, agent, rollout, captures=captures)
learner = PPOLearner(
    transforms=[GAE(...)],
    optimizer=PPOOptimizer(
        agent=agent,
        minibatches=RecurrentRolloutMinibatches(
            sequence_length=32,
            sequences_per_batch=16,
            epochs=4,
        ),
        ...,
    ),
)
```

### SAC

```python
agent = SACAgent(...)
replay = TensorReplayBuffer(
    capacity=1_000_000,
    storage_device="cpu",
    sample_device="cuda",
)
runner = Runner(env, agent.behavior(), replay, captures=())
learner = SACLearner(
    agent=agent,
    transforms=[SACOneStepTarget(gamma=0.99)],
    config=SACConfig(
        sample_spec=ReplaySampleSpec(batch_size=256, steps=1),
        gradient_steps=1,
    ),
)
trainer = Trainer(
    runner,
    replay,
    learner,
    schedule=OffPolicySchedule(
        learning_starts_env_steps=10_000,
        updates_per_env_step=1,
    ),
)
```

The environment runner and core records are shared with PPO. The storage,
target transform, learner, and schedule are intentionally different.

## Specifications and Construction-Time Validation

Field declarations should validate complete recipes before training starts:

```python
@dataclass(frozen=True)
class FieldSpec:
    shape: tuple[int, ...]
    dtype: torch.dtype
    required: bool = True


class ComponentSpec(Protocol):
    requires: Mapping[str, FieldSpec]
    produces: Mapping[str, FieldSpec]
```

Validation should check:

- Every required field has an earlier producer.
- Shapes and dtypes agree at component boundaries.
- A transform does not overwrite a field unless explicitly designated as a
  replacement transform.
- On-policy data contains one policy version.
- A recurrent sampler has state and boundary information.
- The rollout horizon is compatible with sequence length when padding is off.
- A replay window is long enough for burn-in, target steps, and bootstrapping.
- Shared parameters do not appear in incompatible optimizers.

Avoid building all combinations of ready updates. Validate each configured
learner pipeline independently.

## Error Handling and Invariants

Prefer errors that name the stage and component:

```text
prepare/GAE: missing field 'baseline_value'
collect/PPOAgent: log_prob shape [32, 1] does not match env prefix [32]
sample/RecurrentRollout: sequence crosses reset without a reset mask
learn/PPO: rollout policy versions are [41, 42], expected one version
```

Useful runtime assertions can be enabled in development and disabled in hot
production paths after construction-time validation.

## Implementation Status

The runtime currently includes:

- canonical action decisions, environment steps, and explicit captures
- chronological consumable rollouts and persistent lane-aware replay
- flat PPO minibatches, recurrent sequences, and episode-safe replay windows
- ordered reward/value/return transforms with field validation
- shared feed-forward and recurrent PPO optimization
- ordered coarse learning programs and GAIFO discriminator stages
- explicit on-policy and off-policy schedules
- generic Gymnasium collection and specialized CARL self-play collection

Potential future additions include prioritized replay, asynchronous collection,
V-trace or Retrace, and snapshot-backed lazy behavior materialization. These are
not represented by dormant compatibility classes in the current implementation.

## Testing Strategy

### Unit tests

- Rollout insertion order and exact capacity behavior.
- Replay wraparound and direct indexed sampling.
- `terminated` versus `truncated` bootstrap semantics.
- GAE against hand-calculated trajectories.
- N-step targets across termination boundaries.
- Flat minibatch coverage for every epoch.
- Recurrent sequence starts, reset masks, padding, and burn-in.
- Action log probabilities recorded by `act()` equal immediate
  `evaluate_actions()` results before an update.
- Captured and materialized values agree for the same value-function version.
- Policy-version rejection for mixed on-policy rollouts.
- Shared-parameter optimizer validation.
- GAIFO reward materialization uses the discriminator version produced by its
  preceding training stage.

### Integration tests

- PPO learns on a small discrete environment.
- Recurrent PPO solves a partially observable toy environment.
- An off-policy learner samples transition and n-step windows without copying
  the complete replay allocation.
- Checkpoint restore includes model, optimizer, normalizer, replay metadata,
  policy version, and clock.
- A short CARL rollout preserves the expected tensor shapes and recurrent
  likelihood ratios.

### Invariant tests

Several tests should focus on failures rather than learning curves:

- Updating the policy during an unfinished PPO rollout is rejected.
- A recurrent sequence cannot be sampled without boundary information.
- Target updates execute once at their declared point, not once per minibatch.
- Replay sampling does not transfer the complete replay allocation to CUDA.
- Value materialization after a value-function update is rejected.

## Trade-offs

### Explicit learners duplicate some control flow

PPO and SAC will each contain an update loop. That duplication is small and
algorithmically meaningful. The calculations inside the loops remain shared.
Trying to eliminate the loops usually produces a scheduler or graph more
complicated than the duplicated code.

### Optional captures can increase collection work

PPO should retain the log probability already available while selecting an
action. Capturing values may add a successor-value evaluation. Recipes can
instead materialize values before optimization when they preserve model version
and recurrent context. This choice remains visible in the collection and
learning configuration.

### Flat field names are less elegant than nested records

They integrate well with JARL's `TensorBatch` and tensor operations.
Nested views can be added later without changing storage layout.

### Specialized runners reduce apparent uniformity

Uniform record and learner contracts matter more than one universal interaction
loop. CARL should share the parts that are actually common and keep self-play
orchestration explicit.
