import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional

class CrossEntropyLoss2d(nn.Module):
	"""
	taken from https://github.com/bodokaiser/piwise/tree/master/piwise
	"""
	def __init__(self, n_classes, weight=None):
		super().__init__()
		self.n_classes = n_classes
		self.loss = nn.NLLLoss2d(weight)

	def forward(self, outputs, targets):
		return self.loss(F.log_softmax(outputs), targets[0])

	@property
	def output_frame_size(self):
		return self.n_classes

	@property
	def name(self):
		return 'NLLLoss2d'


class NllLoss(nn.Module):
	def __init__(self, output_size, weight:List[float]=None, use_cuda=True):
		super().__init__()
		self.output_size = output_size
		if weight is not None:
			self.weight = torch.Tensor(weight)

			if use_cuda and torch.cuda.is_available():
				self.weight = self.weight.cuda()
		else:
			self.weight = None

	def _transform_logits(self, logits: torch.Tensor) -> torch.Tensor:
		return logits.view(-1, self.output_size)

	def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
		logits = self._transform_logits(logits)
		targets = targets.view(-1)
		#return F.cross_entropy(logits, targets, weight=self.weight)
		return F.nll_loss(logits, targets, weight=self.weight)

	@property
	def name(self):
		return 'NLL Loss'


class LossCombiner(nn.Module):

	def __init__(self, output_size: int, loss_weights: List[List[float]], loss_cls: any):
		super(LossCombiner, self).__init__()
		loss_classes = [loss_cls(output_size, loss_weights[i]) for i in range(len(loss_weights))]
		self.losses = nn.ModuleList(loss_classes)
		self.num_losses = len(loss_classes)

	def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
		# chunk the logits to the different loss functions
		logits = torch.chunk(logits, self.num_losses, dim=1) # tuple with len 20 - each element is a [batch_size, 1, 4] tensor -> for each aspect, one prediction with 4 output values
		targets = torch.chunk(targets, self.num_losses, dim=1) # tuple with len 20 - each element is a [batch_size, 1] tensor -> for each aspect the correct prediction

		# apply losses
		loss: Optional[torch.Tensor] = None
		for i in range(self.num_losses):
			y_hat = logits[i]
			y = targets[i]

			if loss is None:
				loss = self.losses[i](y_hat, y)
			else:
				loss = loss + self.losses[i](y_hat, y)

		return loss

	@property
	def name(self):
		return self.losses[0].name

class MultiHeadNllLoss(nn.Module):
	def __init__(self, output_size: int, n_heads: int, weights:List[List[float]]=None, use_cuda=True):
		super().__init__()
		self.output_size = output_size
		if weights is not None:
			self.weight = torch.Tensor(weights)

			if use_cuda and torch.cuda.is_available():
				self.weight = self.weight.cuda()
		else:
			self.weight = None

	def _transform_logits(self, logits: torch.Tensor) -> torch.Tensor:
		return logits.view(-1, self.output_size)

	def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
		logits = self._transform_logits(logits)
		targets = targets.view(-1)
		return F.nll_loss(logits, targets, weight=self.weight)

	@property
	def name(self):
		return 'NLL Loss'
	



class MSELoss(nn.Module):
	def __init__(self):
		super().__init__()
		self.loss = nn.MSELoss()

	def forward(self, outputs, targets):
		return self.loss(outputs, targets)

	@property
	def output_frame_size(self):
		return 1