Layed out here an example of how I want ``JARL`` to be used. It is intended to be...

1. Highly modular with limited "local complexity"
2. Algorithm-agnostic in its basic structure
3. Extensible to a wide array of known and novel algorithms

Consider we have defined our environment and policy:

```python
# initialize PyTorch gym env
env = gym.make("LunarLander-v2")
env = TorchGymEnv(env)

device = "cuda"  # gpu compatible

# initialize policy
policy = CategoricalPolicy(
    head=FlattenEncoder(),
    body=MLP(64, 64),
    env=env  # optional env argument for implicit initialization
).to(device)
```

For instance, Soft Actor-Critic (SAC), which involves several modules that interoperate to perform parameter updates, may be implemented as follows:

```python
# initialize q-networks & targets
qnet_1 = QNet(
    head=FlattenEncoder(),
    body=MLP(32, 32)
).initialize(env).to(device)

qnet_2 = QNet(
    head=FlattenEncoder(),
    body=MLP(32, 32)
).initialize(env).to(device)

targ_1, targ_2 = qnet_1.copy(), qnet_2.copy()

# create update loop
loop = TrainLoop(env=env, policy=policy, buffer=Buffer())

# add & fill update block
# TODO: DataModifier and ModuleUpdate initializations
block = UpdateBlock()
block.set_batch_sampler(BatchSampler())
block.add_batch_modifier(DataModifier())  # compute targets
block.add_module_update(ModuleUpdate())   # update Q funcs
block.add_module_update(ModuleUpdate())   # update policy
block.add_module_update(ModuleUpdate())   # polyak update targets
loop.add_block(block)

# run training
loop.run(steps=int(2**20))
```

Proximal Policy Optimization (PPO) may be implemented as:

```python
critic = Critic(
    head=FlattenEncoder(),
    body=MLP(32, 32)
).initialize(env).to(device)

block = UpdateBlock()

# modify buffer prior to sampling
block.add_buffer_modifiers(
    ComputeValues(critic),
    ComputeAdvantage(lmbda=0.99, gamma=0.95)
)

# set sampler used for updates
block.set_batch_sampler(BatchSampler(64))

# add module updates for policy and critic
block.add_module_updates(
    ClippedPolicyLoss(policy, critic, **pol_params),
    CriticMSELoss(critic, optimizer=Optimizer(Adam, **opt_params))
)

# block.add_module_updates(PPOLoss(policy, critic, **params))

loop.add_block(block)
loop.run(steps=int(2**20))
```

Generative Adversarial Imitation from Observation may be implemented as a superset of PPO. Predefined ``PPOBlock()`` can be plugged in after a block dictating the update of a discriminator.

```python
from blocks.ppo import PPOBlock

discrim = Discriminator(
    head=FlattenEncoder(),
    body=MLP(64, 64)
).initialize(env).to(device)

block_0 = UpdateBlock()
block_0.set_batch_sampler(ExpertSampler())  # sample from expert/policy pairs
block_0.add_module_update(ModuleUpdate())   # update discriminator
loop.add_block(block_0)

# reuse established PPO block
block_1 = PPOBlock(policy, critic)
block_1.add_data_modifier(DataModifier(), at=0)  # reward shaping via discriminator
loop.add_block(block_1)

loop.run(steps=int(2**20))
```