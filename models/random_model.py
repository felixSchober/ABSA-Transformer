import logging
from typing import Optional, List
import torch
import torch.nn as nn

from misc.run_configuration import RunConfiguration, OutputLayerType
from models.transformer.encoder import TransformerEncoder
from models.output_layers import CommentWiseConvLogSoftmax, CommentWiseSumLogSoftmax, CommentWiseConvLinearLogSoftmax

class RandomModel(nn.Module):
	"""description of class"""

	
	model_size: int
	target_size: int
	num_taggers: int
	hyperparameters: RunConfiguration

	def __init__(self, hyperparameters: RunConfiguration, target_size: int, num_taggers: int, names: List[str]=[]):
		super(RandomModel, self).__init__()

		assert hyperparameters.model_size > 0
		assert target_size > 0
		assert num_taggers > 0

		self.hyperparameters = hyperparameters
		self.logger = logging.getLogger(__name__)

		self.logger.warn('You are using a random classifier. Do not expect miracles ;)')
		print('################################# RANDOM MODE #################################')
		print('#########                                                              ########')
		print('#########                                                              ########')
		print('######### You are using a random classifier. Do not expect miracles ;) ########')
		print('#########    This classifier will stop evaluation after one epoch.     ########')
		print('#########                                                              ########')
		print('#########                                                              ########')
		print('################################# RANDOM MODE #################################')

		self.model_size = self.hyperparameters.model_size
		self.target_size = target_size
		self.num_taggers = num_taggers
		self.names = names
		hyperparameters.num_epochs = 1


	def forward(self, x: torch.Tensor, *args) -> torch.Tensor:
		if self.hyperparameters.task == 'ner':
			return torch.rand((x.shape[0], x.shape[1], self.target_size), requires_grad=True)
		return torch.rand((x.shape[0], self.num_taggers, self.target_size), requires_grad=True)

	def predict(self, x: torch.Tensor, *args) -> torch.Tensor:
		if self.hyperparameters.task == 'ner':
			return torch.randint(0, self.target_size, (x.shape[0], x.shape[1]))

		return torch.randint(0, self.target_size, (x.shape[0], self.num_taggers))

