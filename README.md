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

## PPO Example

Given an environment, policy, and value function, a PPO training setup can be
assembled from JARL's collection, storage, sampling, and learning components:

```python
from torch.optim import Adam

from jarl.collect import LogProbCapture, PolicyVersionCapture, Runner, ValueCapture
from jarl.learn import OptimizerStep, PPOConfig, PPOLoss, Update, unique_parameters
from jarl.runtime import OnPolicySchedule, Trainer
from jarl.sample import RolloutMinibatches
from jarl.store import RolloutBuffer
from jarl.transform import GAE

rollout = RolloutBuffer(horizon=128, num_envs=env.n_envs, device="cuda")
runner = Runner(
    env,
    policy,
    rollout,
    captures=(
        LogProbCapture(),
        PolicyVersionCapture(policy),
        ValueCapture(value_function),
    ),
)

parameters = unique_parameters((policy, value_function))
optimizer = Adam(parameters, lr=2.5e-4)
learner = Update(
    transforms=(GAE(gamma=0.99, lambda_=0.95),),
    sampler=RolloutMinibatches(batch_size=256, epochs=4),
    loss=PPOLoss(
        policy,
        value_function,
        PPOConfig(clip=0.2, entropy_coef=0.01),
    ),
    optimizer_step=OptimizerStep(
        (policy, value_function),
        optimizer,
        max_grad_norm=0.5,
    ),
    section="PPO",
)

trainer = Trainer(runner, rollout, learner, OnPolicySchedule())
trainer.run(total_env_steps=1_000_000)
```
