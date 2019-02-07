import torch
import torch.nn as nn
import torch.nn.functional as F

class SoftmaxOutputLayer(nn.Module):

	def __init__(self, hidden_size: int, output_size: int):
		"""Projects the output of a model like the transformer encoder to a desired shape so that it is possible to perform classification.
		This is done by projecting the output of the model (e.g. size 300 per word) down to the ammount of classes
		
		Arguments:
			hidden_size {int} -- output of the transformer encoder (d_model)
			output_size {int} -- number of classes
		"""
		super(SoftmaxOutputLayer, self).__init__()
		self.output_size = output_size
		self.output_projection = nn.Linear(hidden_size, output_size)

	def forward(self, x, *args):
		logits = self.output_projection(x)
		probs = F.log_softmax(logits, dim=-1)

		return probs


class SoftmaxOutputLayerWithCommentWiseClass(nn.Module):

	def __init__(self, hidden_size: int, output_size: int, name: str = None):
		"""Projects the output of a model like the transformer encoder to a desired shape so that it is possible to perform classification.
		This is done by projecting the output of the model (e.g. size 300 per word) down to the ammount of classes
		
		Arguments:
			hidden_size {int} -- output of the transformer encoder (d_model)
			output_size {int} -- number of classes
		"""
		super(SoftmaxOutputLayerWithCommentWiseClass, self).__init__()
		self.output_size = output_size
		self.output_projection = nn.Linear(hidden_size, output_size)
		self.name = name if name is not None else 'NotSet'

	def forward(self, x: torch.Tensor, mask: torch.Tensor =None, *args):
		logits = self.output_projection(x)

		# apply mask so that the paddings don't contribute anything
		if mask is not None:
			# transform mask so that it can be applied to the logits.
			# the logits will be in the form of [batch_size, num_words, num_classes] (e.g. 80, 679, 4)
			# the mask will be in the form of   [batch_size, 1, num_words] (e.g. 80, 1, 679)
			# First, transform the mask to [batch_size, num_words] and then to [batch_size, num_words, 1]
			transformed_mask = mask.squeeze(1).unsqueeze(-1)
			logits.masked_fill(transformed_mask == 0, 0)

		logits = torch.sum(logits, dim=1)
		probs = F.log_softmax(logits, dim=-1)

		return probs

class ConvSoftmaxOutputLayerWithCommentWiseClass(SoftmaxOutputLayerWithCommentWiseClass):

	def __init__(self, hidden_size: int, output_size: int):
		"""Projects the output of a model like the transformer encoder to a desired shape so that it is possible to perform classification.
		This is done by projecting the output of the model (e.g. size 300 per word) down to the ammount of classes
		
		Arguments:
			hidden_size {int} -- output of the transformer encoder (d_model)
			output_size {int} -- number of classes
		"""
		super(ConvSoftmaxOutputLayerWithCommentWiseClass, self).__init__(hidden_size, output_size)
		
		self.comment_projection = nn.Conv1d()

	def forward(self, x: torch.Tensor, mask: torch.Tensor =None, *args):
		logits = self.output_projection(x)

		# apply mask so that the paddings don't contribute anything
		if mask is not None:
			# transform mask so that it can be applied to the logits.
			# the logits will be in the form of [batch_size, num_words, num_classes] (e.g. 80, 679, 4)
			# the mask will be in the form of   [batch_size, 1, num_words] (e.g. 80, 1, 679)
			# First, transform the mask to [batch_size, num_words] and then to [batch_size, num_words, 1]
			transformed_mask = mask.squeeze(1).unsqueeze(-1)
			logits.masked_fill(transformed_mask == 0, 0)

		logits = torch.sum(logits, dim=1)
		probs = F.log_softmax(logits, dim=-1)

		return probs

class OutputLayer(nn.Module):

	def __init__(self, hidden_size: int, output_size: int):
		"""Projects the output of a model like the transformer encoder to a desired shape so that it is possible to perform classification for example
		
		Arguments:
			hidden_size {int} -- output of the transformer encoder (d_model)
			output_size {int} -- number of classes
		"""
		super(OutputLayer, self).__init__()
		self.output_size = output_size
		self.output_projection = nn.Linear(hidden_size, output_size)

	def forward(self, x, *args):
		logits = self.output_projection(x)
		return logits

	def predict(self, x):
		logits = self.forward(x)
		probs = F.softmax(logits, -1)
		_, predictions = torch.max(probs, dim=-1)
		return predictions