# JARL (WIP)

JARL is written to be highly modular and allow for rapid prototyping of different RL algorithms.
Eventually, many existing algorithms will be implemented by default within JARL. Their core components will therefore be available to rearrange and refactor into more novel approaches.
JARL utilizes a few core proprietary objects...

### 1. MultiTensor

A MultiTensor is just a nested Python dictionary of PyTorch tensors (with dot attribute access). It is indexable in the same way a tensor is, for instance:

```python
import torch as th

data = MultiTensor(dict(
    a=th.rand((5,3)),
    b=th.rand((5,)),
    c=dict(
        d=th.rand((5, 2, 4))
    )
))

# data[:3].a == data.a[:3]
# data[:3].c.d == data.c.d[:3]
# and so on
```
