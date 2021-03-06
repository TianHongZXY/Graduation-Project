import torch.nn as nn
import torch.optim as optim


class Optimizer(object):
    def __init__(self,
                 params,
                 lr,
                 lr_decay=1.0,
                 weight_decay=0.0,
                 max_grad_norm=None):
        self.parameters = params
        self.lr = lr
        self.lr_decay = lr_decay
        self.weight_decay = weight_decay
        self.max_grad_norm = max_grad_norm
        self.optimizer = None
        self._step = 0

    def update_lr(self, epoch):
        self.lr = self.lr * self.lr_decay ** epoch
        for param in self.optimizer.param_groups:
            param['lr'] = self.lr

    def step(self):
        if self.max_grad_norm and self.max_grad_norm > 0:
            nn.utils.clip_grad_norm_(self.parameters, self.max_grad_norm)
        self.optimizer.step()
        self._step += 1

    def zero_grad(self, set_to_none=False):
        self.optimizer.zero_grad(set_to_none=set_to_none)

    def rate(self):
        return self.lr

    def get_global_step(self):
        return self._step


class AdamOptimizer(Optimizer):
    def __init__(self, params,
                 lr=1e-3,
                 lr_decay=1.0,
                 weight_decay=0.0,
                 max_grad_norm=None,
                 betas=(0.9, 0.999),
                 eps=1e-8):
        super(AdamOptimizer, self).__init__(params, lr, lr_decay, weight_decay, max_grad_norm)
        self.optimizer = optim.Adam(params, lr=self.lr, weight_decay=self.weight_decay, betas=betas, eps=eps)
        self.param_groups = self.optimizer.param_groups


class SGDOptimizer(Optimizer):
    def __init__(self, params,
                 lr,
                 lr_decay=1.0,
                 weight_decay=0.0,
                 max_grad_norm=None,
                 momentum=0):
        super(SGDOptimizer, self).__init__(params, lr, lr_decay, weight_decay, max_grad_norm)
        self.optimizer = optim.SGD(params, lr=self.lr, weight_decay=self.weight_decay, momentum=momentum)
        self.param_groups = self.optimizer.param_groups


class RMSpropOptimizer(Optimizer):
    def __init__(self, params,
                 lr,
                 lr_decay=1.0,
                 weight_decay=0.0,
                 max_grad_norm=None,
                 alpha=0.99,
                 eps=1e-8,
                 momentum=0):
        super(RMSpropOptimizer, self).__init__(params, lr, lr_decay, weight_decay, max_grad_norm)
        self.optimizer = optim.RMSprop(params, lr=self.lr, weight_decay=self.weight_decay,
                                       alpha=alpha, eps=eps, momentum=momentum)
        self.param_groups = self.optimizer.param_groups


class NoamOptimWrapper:
    """Transformer's optimizer. An optim wrapper that implements warmup learning rate."""
    def __init__(self, model_size, factor, warmup, optimizer):
        self.optimizer = optimizer
        self._step = 0
        self.warmup = warmup
        self.factor = factor
        self.model_size = model_size
        self._rate = 0

    def step(self):
        """Update parameters and rate"""
        self._step += 1
        rate = self.rate()
        for p in self.optimizer.param_groups:
            p['lr'] = rate
        self._rate = rate
        self.optimizer.step()

    def rate(self, step=None):
        """
        Implement `lrate` = d_model^(-0.5) * min(step_num^(-0.5), step_num * warmup_steps^(-1.5))
        lr ?????????d_model^(-0.5) * warmup_steps^(-0.5) = (d_model * warmup_steps)^(-0.5)
        """
        if step is None:
            step = self._step
        return self.factor * (self.model_size ** (-0.5) *
                min(step ** (-0.5), step * self.warmup ** (-1.5)))

    def zero_grad(self, set_to_none=False):
        self.optimizer.zero_grad(set_to_none=set_to_none)

    def get_global_step(self):
        return self._step

