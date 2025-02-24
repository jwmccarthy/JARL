from torch.nn.utils import clip_grad_norm_

from itertools import chain


class Optimizer:

    def __init__(self, optimizer, max_grad_norm=0.5, **op_kwargs):
        self.optimizer = optimizer
        self.op_kwargs = op_kwargs
        self.max_grad_norm = max_grad_norm

    def build(self, modules):
        self.params = chain(*[m.parameters() for m in modules])
        self.optimizer = self.optimizer(self.params, **self.op_kwargs)
        return self
    
    def update(self, loss):
        self.optimizer.zero_grad()
        loss.backward()

        # clip gradients
        if self.max_grad_norm:
            clip_grad_norm_(self.params, self.max_grad_norm)

        self.optimizer.step()