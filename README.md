# JARL (WIP)

Still under active dev. Pivoting for now to develop proprietary on-GPU simulations for improved speed. In its current state, JARL is fit to implement and run a wide variety of algorithms.

JARL is written to be highly modular and allow for rapid prototyping of different RL algorithms.
Eventually, many existing algorithms will be implemented by default within JARL. Their core components will therefore be available to rearrange and refactor into more novel approaches.
JARL utilizes a few core proprietary objects...

### ```MultiTensor```

A ```MultiTensor``` is subclasses the Python dictionary and contains same-size PyTorch tensors (with dot attribute access). It is indexable in the same way a tensor is, for instance:

```python
import torch as th

data = MultiTensor(
    a=th.rand((5,3)),
    b=th.rand((5,)),
)

th.equal(data[:3].a, data.a[:3])  # => True
```

This is useful for the construction of easy-to-manipulate replay buffers and the passing of complex information between modules.

### ```TrainGraph```

The ```TrainGraph``` is the core of the RL training loop in JARL. It relies on the following components:

#### 1. ```Sampler```

A ```Sampler``` maps the ```MultiTensor``` served by the replay buffer to a generator of ```MultiTensor```, each containing a subset of the original data.

#### 2. ```ModuleUpdate```

A ```ModuleUpdate``` contains a set of ```torch.nn.Module``` derivative classes, and takes as input a ```MultiTensor``` of the prerequisite information for its loss calculation. Each ```ModuleUpdate``` also has a set of required keys that should be present in its input. ```ModuleUpdate.ready(t)``` indicates whether the module is ready to update at a given timestep, which is determined by its update frequency (```freq``` parameter).

#### 3. ```DataModifier```

A ```DataModifier``` takes a ```MultiTensor``` as input and returns the same object plus some modifications (altered contents, additional information, etc.) Similar to ```ModuleUpdate```, each ```DataModifier``` has a set of required keys, as well as a set of keys that it produces in its output.

#### Utilizing ```TrainGraph```

With these 3 module types in mind, we can construct a ```TrainGraph```. We initialize it with a sampler and a list of ```ModuleUpdate```, and then iteratively add the ```DataModifier``` we need via the ```TrainGraph.add_modifier()``` method.

```python
graph = (
    TrainGraph(BatchSampler(), PPOUpdate(policy, critic))
    .add_modifier(ComputeValues(critic))
    .add_modifier(ComputeAdvantages())
    .add_modifier(ComputeReturns())
    .add_modifier(ComputeLogProbs(policy))
)
```

At the end we call 

```python
graph.compile()
```

which does the following:

1. Perform a topological sort of all ```DataModifier``` based on their dependencies (required and produced keys)
2. For each combination of ```ModuleUpdate``` in the graph, extract the necessary ```DataModifier``` sequence
3. Create a function composition from each sequence and store indexed by binary mask of module combination

From here, we can have updates of varying frequency that run only their necessary prerequisite ```DataModifier``` when they're ready. This is wildly overkill for simple algorithms, but it is highly generalizable, customizable, and makes intuitive sense for algorithms with multiple updates that operate at different frequencies. For instance, SAC may utilize different update timings for its Q updates, policy updates, and target network Polyak updates.

Notably, on-policy and off-policy algorithms are only distinguished in JARL by the size of the replay buffer in relation to the update timings. For on-policy algorithms, the circular replay buffer will have a maximum capacity equal to the update frequency. Off-policy algorithms will have much larger buffers than the update frequency.

I realize this is probably an overly complicated way to go about things, but there are so many frameworks already and I wanted to do something new!
