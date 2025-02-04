# JARL (WIP)

JARL is written to be highly modular and allow for rapid prototyping of different RL algorithms.
Eventually, many existing algorithms will be implemented by default within JARL. Their core components will therefore be available to rearrange and refactor into more novel approaches.
JARL utilizes a few core proprietary objects...

### ```MultiTensor```

A ```MultiTensor``` is just a nested Python dictionary of PyTorch tensors (with dot attribute access). It is indexable in the same way a tensor is, for instance:

```python
import torch as th

data = MultiTensor(dict(
    a=th.rand((5,3)),
    b=th.rand((5,)),
    c=dict(
        d=th.rand((5, 2, 4))
    )
))

all(data[:3].a == data.a[:3])     # => True
all(data[:3].c.d == data.c.d[:3]) # => True
```

This is useful for the construction of easy-to-manipulate replay buffers and the passing of complex information between modules.

### ```TrainGraph```

The ```TrainGraph``` is the core of the RL training loop in JARL. It relies on the following components:

#### 1. ```Sampler```

A ```Sampler``` maps the ```MultiTensor``` served by the replay buffer to a generator ```MultiTensor```s, each containing a subset of the original data.

#### 2. ```ModuleUpdate```

A ```ModuleUpdate``` contains a set of ```torch.nn.Module``` derivative classes, and takes as input a ```MultiTensor``` that contains the prerequisite information for its loss calculation. Each ```ModuleUpdate``` also has a set of required keys that should be present in its input.

#### 3. ```DataModifier```

A ```DataModifier``` takes a ```MultiTensor``` as input and returns the same object plus some modifications (altered contents, additional information, etc.) Similar to ```ModuleUpdate```, each ```DataModifier``` has a set of required keys, as well as a set of keys that it produces in its output.

With these 3 module types in mind, we can construct a ```TrainGraph```. We initialize it with a sampler and a list of ```ModuleUpdate```s, and then iteratively add the ```DataModifier```s we need via the ```TrainGraph.add_modifier()``` method.

```python
graph = (
    TrainGraph(BatchSampler(), PPOUpdate(policy, critic))
    .add_modifier(ComputeValues(critic))
    .add_modifier(ComputeAdvantages())
    .add_modifier(ComputeReturns())
    .add_modifier(ComputeLogProbs())
)
```

At the end we call 

```python
graph.compile()
```

which does the following:

#### 1.
