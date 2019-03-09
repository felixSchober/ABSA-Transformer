import torch

from misc.run_configuration import RunConfiguration, LearningSchedulerType, OptimizerType

class NoamOptimizer(torch.optim.Optimizer):
    "Noam Optim wrapper that implements rate."
    def __init__(self, model_size: int, factor: float, warmup: float, base_optimizer: torch.optim.Optimizer):

        self.optimizer = base_optimizer
        self._step = 0
        self.warmup = warmup
        self.factor = factor
        self.model_size = model_size
        self._rate = 0
        
    def step(self, *args) -> None:
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

class OptimizerWrapper(torch.optim.Optimizer):
    def __init__(self, base_optimizer: torch.optim.Optimizer):
        self.optimizer = base_optimizer

    def step(self, *args) -> None:
        self.optimizer.step()

    def zero_grad(self):
        return self.optimizer.zero_grad()

    def rate(self, step: int = None) -> float:
        return 0.0

        
def get_optimizer(model, hp: RunConfiguration) -> torch.optim.Optimizer:

	if hp.optimizer_type == OptimizerType.Adam:
		optimizer = torch.optim.Adam(model.parameters(),
                            lr=hp.learning_rate,
                            betas=(hp.adam_beta1, hp.adam_beta2),
                            eps=hp.adam_eps,
							weight_decay=hp.adam_weight_decay,
							amsgrad=hp.adam_amsgrad)

	elif hp.optimizer_type == OptimizerType.SGD:
		optimizer = torch.optim.SGD(
							model.parameters(),
							lr=hp.learning_rate,
							momentum=hp.sgd_momentum,
							dampening=hp.sgd_dampening,
							nesterov=hp.sgd_nesterov
		)

	elif hp.optimizer_type == OptimizerType.RMS_PROP:
		optimizer = torch.optim.RMSprop(
										model.parameters(),
										lr=hp.learning_rate,
										alpha=hp.rmsprop_alpha,
										eps=hp.rmsprop_eps,
										weight_decay=hp.rmsprop_weight_decay,
										centered=hp.rmsprop_centered,
										momentum=hp.rmsprop_momentum
		)

	elif hp.optimizer_type == OptimizerType.AdaBound:
		from adabound import AdaBound
		optimizer = AdaBound(model.parameters(), lr=hp.learning_rate, final_lr=hp.adabound_finallr)

	# elif hp.learning_rate_type == LearningSchedulerType.Adadelta:
	# 	optimizer = torch.optim.Adadelta(model.parameters(), 
	# 	lr=hp.learning_rate)

	return wrap_optimizer(hp, optimizer)


def wrap_optimizer(hp: RunConfiguration, optimizer: torch.optim.Optimizer):
	if hp.learning_rate_scheduler_type == LearningSchedulerType.NoSchedule:
		return OptimizerWrapper(optimizer)

	elif hp.learning_rate_scheduler_type == LearningSchedulerType.Noam:
		return NoamOptimizer(hp.model_size,
                            hp.noam_learning_rate_factor,
                            hp.noam_learning_rate_warmup,
                            optimizer)

	elif hp.learning_rate_scheduler_type == LearningSchedulerType.Exponential:
		return torch.optim.lr_scheduler.ExponentialLR(
			optimizer,
			gamma=hp.exponentiallr_gamma
        ) 

	

