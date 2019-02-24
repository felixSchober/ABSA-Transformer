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

	def predict(self, x: torch.Tensor, mask: torch.Tensor =None, *args):
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
		probs = F.softmax(logits, dim=-1)

		return probs

class ConvSoftmaxOutputLayerWithCommentWiseClass(nn.Module):

	def __init__(self, model_size: int, kernel_size: int, stride: int, padding: str, num_filters: int, output_size: int, sentence_lenght: int, name: str = None):
		"""Projects the output of a model like the transformer encoder to a desired shape so that it is possible to perform classification.
		This is done by projecting the output of the model (e.g. size 300 per word) down to the ammount of classes
		
		Arguments:
			hidden_size {int} -- output of the transformer encoder (d_model)
			output_size {int} -- number of classes
		"""
		super(ConvSoftmaxOutputLayerWithCommentWiseClass, self).__init__()
		self.name = name if name is not None else 'NotSet'
		self.output_size = output_size

		self.conv_out = ((sentence_lenght + 2 * padding - 1 * (kernel_size - 1) - 1) // stride) + 1
		self.conv = nn.Sequential(
			nn.Conv2d(1, num_filters, (kernel_size, model_size), stride, padding),
			nn.ReLU()
		)
		self.pooling = nn.MaxPool2d((self.conv_out, 1), stride=stride)
		self.output_projection = nn.Linear(num_filters, output_size)

	def forward(self, x: torch.Tensor, mask: torch.Tensor =None, *args):
		x = x.unsqueeze(1) 					# [batch_size, num_words, model_size] -> e.g. [12, 100, 300] -> [batch_size, 1, num_words, model_size]

		x = self.conv(x)					# [batch_size, 1, num_words, model_size] -> [batch_size, num_filters, num_words - padding, 1] e.g. [12, 300, 96, 1]
		x = self.pooling(x)					# [batch_size, num_filters, num_words - padding, 1] -> [batch_size, num_filters, 1, 1]
		x = x.squeeze().squeeze()
		logits = self.output_projection(x)	# [batch_size, num_filters] -> [batch_size, classes] e.g. [12, 4]

		probs = F.log_softmax(logits, dim=-1)
		return probs

	def predict(self, x: torch.Tensor, mask: torch.Tensor=None, *args):
		x = x.unsqueeze(1) 					# [batch_size, num_words, model_size] -> e.g. [12, 100, 300] -> [batch_size, 1, num_words, model_size]

		x = self.conv(x)					# [batch_size, 1, num_words, model_size] -> [batch_size, num_filters, num_words - padding, 1] e.g. [12, 300, 96, 1]
		x = self.pooling(x)					# [batch_size, num_filters, num_words - padding, 1] -> [batch_size, num_filters, 1, 1]
		x = x.squeeze().squeeze()
		logits = self.output_projection(x)	# [batch_size, num_filters] -> [batch_size, classes] e.g. [12, 4]

		probs = F.softmax(logits, dim=-1)
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