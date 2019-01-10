import torch
import torch.nn as nn
import torch.nn.functional as F

from misc.run_configuration import RunConfiguration

class NoamOptimizer(object):
    "Noam Optim wrapper that implements rate."
    def __init__(self, model_size: int, factor: float, warmup: float, base_optimizer: torch.optim.Optimizer):

        self.optimizer = base_optimizer
        self._step = 0
        self.warmup = warmup
        self.factor = factor
        self.model_size = model_size
        self._rate = 0
        
    def step(self) -> None:
        "Update parameters and rate"
        self._step += 1
        rate = self.rate()
        for p in self.optimizer.param_groups:
            p['lr'] = rate
        self._rate = rate
        self.optimizer.step()

    def zero_grad(self):
        return self.optimizer.zero_grad()
        
    def rate(self, step: int = None) -> float:
        "Implement `lrate` above"
        if step is None:
            step = self._step
        return self.factor * \
            (self.model_size ** (-0.5) *
            min(step ** (-0.5), step * self.warmup ** (-1.5)))


        
def get_default_optimizer(model, hyperparameters: RunConfiguration):
    adam = torch.optim.Adam(model.parameters(),
                            lr=hyperparameters.learning_rate,
                            betas=(hyperparameters.optim_adam_beta1, hyperparameters.optim_adam_beta2),
                            eps=1e-9)
    return NoamOptimizer(hyperparameters.model_size,
                         hyperparameters.learning_rate_factor,
                         hyperparameters.learning_rate_warmup,
                         adam)