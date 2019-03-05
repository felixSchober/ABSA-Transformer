import logging
from typing import Optional, List
import torch
import torch.nn as nn

from misc.run_configuration import RunConfiguration
from models.transformer.encoder import TransformerEncoder
from models.output_layers import CommentWiseConvLogSoftmax, CommentWiseSumLogSoftmax, CommentWiseConvLinearLogSoftmax

class JointAspectTagger(nn.Module):
	"""description of class"""

	encoder: TransformerEncoder
	taggers: nn.ModuleList
	model_size: int
	target_size: int
	num_taggers: int
	logger: logging.RootLogger
	hyperparameters: RunConfiguration

	def __init__(self, transformerEncoder: TransformerEncoder, hyperparameters: RunConfiguration, target_size: int, num_taggers: int, names: List[str]=[]):
		super(JointAspectTagger, self).__init__()

		assert hyperparameters.model_size > 0
		assert target_size > 0
		assert num_taggers > 0

		self.hyperparameters = hyperparameters
		self.encoder = transformerEncoder
		self.logger = logging.getLogger('pre_training')

		self.model_size = self.hyperparameters.model_size
		self.target_size = target_size
		self.num_taggers = num_taggers
		self.names = names
		
		self.taggers = self.initialize_aspect_taggers()
		self.logger.debug(f"{self.num_taggers} initialized")
		
		self.logger.debug(f"Initilize parameters with nn.init.xavier_uniform_")
		for p in self.parameters():
			if p.dim() > 1:
				nn.init.xavier_uniform_(p)
		self.logger.debug(f"Tagger initialized")


	def initialize_aspect_taggers(self):
		taggers = []
		hp = self.hyperparameters
		names = self.names if len(self.names) > 0 else range(self.num_taggers)
		for n in names:

			if hp.output_layer_type == 'conv':
				tagger = CommentWiseConvLogSoftmax(
												hp.model_size,
												hp.output_conv_kernel_size,
												hp.output_conv_stride,
												hp.output_conv_padding, 
												hp.output_conv_num_filters,
												self.target_size,
												hp.clip_comments_to,
												'Apsect ' + n
												)
			elif hp.output_layer_type == 'convLinear':
				tagger = CommentWiseConvLinearLogSoftmax(
												hp.model_size,
												hp.output_conv_kernel_size,
												hp.output_conv_stride,
												hp.output_conv_padding, 
												hp.output_conv_num_filters,
												self.target_size,
												hp.clip_comments_to,
												1024,
												'Apsect ' + n
												)
			else:
				tagger = CommentWiseSumLogSoftmax(hp, self.model_size, self.target_size, 'Apsect ' + n)

			taggers.append(tagger)
		return nn.ModuleList(taggers)

	def forward(self, x: torch.Tensor, *args) -> torch.Tensor:
		result = self.encoder(x, *args) # result will be [batch_size, num_words, model_size]

		output: torch.Tensor = None

		# provide the result to each aspect tagger
		for _, aspect_tagger in enumerate(self.taggers):
			tagging_result = aspect_tagger(result, *args)
			tagging_result = torch.unsqueeze(tagging_result, 1)

			if output is None:
				output = tagging_result
			else:
				output = torch.cat((output, tagging_result), 1)

		# output is now [batch_size, num_aspects, 4] (12, 20, 4)
		# to reduce it to [batch_size, 4] we need to get the max 
		return output

	def predict(self, x: torch.Tensor, *args) -> torch.Tensor:
		result = self.encoder(x, *args) # result will be [batch_size, num_words, model_size]
		output: torch.Tensor = None

		# provide the result to each aspect tagger
		for _, aspect_tagger in enumerate(self.taggers):
			tagging_result = aspect_tagger.predict(result, *args)
			_, tagging_result = torch.max(tagging_result, dim=-1) 
			if output is None:
				if len(tagging_result.shape) < 2:
					output = tagging_result.unsqueeze(1)
				else:
					output = tagging_result
			else:
				output = torch.cat((output, tagging_result.unsqueeze(1)), 1)
		return output 

