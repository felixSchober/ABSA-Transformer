import torch
import torch.nn as nn
import torch.nn.functional as F

from misc.run_configuration import RunConfiguration

class LogSoftmaxOutputLayer(nn.Module):

	def __init__(self, hidden_size: int, output_size: int):
		"""Projects the output of a model like the transformer encoder to a desired shape so that it is possible to perform classification.
		This is done by projecting the output of the model (e.g. size 300 per word) down to the ammount of classes.
		This means that this layer predicts on the word level.
		Uses the log softmax
		
		Arguments:
			hidden_size {int} -- output of the transformer encoder (d_model)
			output_size {int} -- number of classes
		"""
		super(LogSoftmaxOutputLayer, self).__init__()
		self.output_size = output_size
		self.output_projection = nn.Linear(hidden_size, output_size)

	def forward(self, x, *args):
		logits = self.output_projection(x)
		probs = F.log_softmax(logits, dim=-1)

		return probs

	def predict(self, x):
		logits = self.output_projection(x)
		probs = F.softmax(logits, -1)
		_, predictions = torch.max(probs, dim=-1)
		return predictions

class SoftmaxOutputLayer(nn.Module):

	def __init__(self, hidden_size: int, output_size: int):
		"""Projects the output of a model like the transformer encoder to a desired shape so that it is possible to perform classification.
		This is done by projecting the output of the model (e.g. size 300 per word) down to the ammount of classes.
		This means that this layer predicts on the word level
		Uses the log normal softmax
		
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


class CommentWiseSumLogSoftmax(nn.Module):

	def __init__(self, hp: RunConfiguration, hidden_size: int, output_size: int, name: str = None):
		"""Projects the output of a model like the transformer encoder to a desired shape so that it is possible to perform classification.
		This is done by projecting the output of the model (e.g. size 300 per word) down to the ammount of classes.
		This uses the sum operation to predict on comment level.
		
		Arguments:
			hidden_size {int} -- output of the transformer encoder (d_model)
			output_size {int} -- number of classes
		"""
		super(CommentWiseSumLogSoftmax, self).__init__()
		self.output_size = output_size
		
		self.output_projection = nn.Linear(hidden_size, output_size)
		self.dropout = nn.Dropout(hp.last_layer_dropout)
		self.name = name if name is not None else 'NotSet'

	def forward(self, x: torch.Tensor, mask: torch.Tensor =None, *args):
		logits = self.output_projection(x)
		logits = self.dropout(logits)

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
		probs = F.log_softmax(logits, dim=-1)

		return probs

class CommentWiseConvLogSoftmax(nn.Module):

	def __init__(self, model_size: int, kernel_size: int, stride: int, padding: str, num_filters: int, output_size: int, sentence_lenght: int, name: str = None):
		"""Projects the output of a model like the transformer encoder to a desired shape so that it is possible to perform classification.
		This is done by projecting the output of the model (e.g. size 300 per word) down to the ammount of classes.
		This uses convolutions to predict on comment level.


		Arguments:
			hidden_size {int} -- output of the transformer encoder (d_model)
			output_size {int} -- number of classes
		"""
		super(CommentWiseConvLogSoftmax, self).__init__()
		self.name = name if name is not None else 'NotSet'
		self.output_size = output_size

		self.conv_out = ((sentence_lenght + 2 * padding - 1 * (kernel_size - 1) - 1) // stride) + 1
		self.conv = nn.Sequential(
			nn.Conv2d(1, num_filters, (kernel_size, model_size), stride, padding),
			nn.ReLU(),
			nn.MaxPool2d((self.conv_out, 1), stride=stride)
		)
		self.output_projection = nn.Linear(num_filters, output_size)

	def forward(self, x: torch.Tensor, mask: torch.Tensor =None, *args):
		x = x.unsqueeze(1) 					# [batch_size, num_words, model_size] -> e.g. [12, 100, 300] -> [batch_size, 1, num_words, model_size]

		x = self.conv(x)					# [batch_size, 1, num_words, model_size] -> [batch_size, num_filters, num_words - padding, 1] e.g. [12, 300, 96, 1]
		# x = self.pooling(x)					# [batch_size, num_filters, num_words - padding, 1] -> [batch_size, num_filters, 1, 1]
		x = x.squeeze().squeeze()
		logits = self.output_projection(x)	# [batch_size, num_filters] -> [batch_size, classes] e.g. [12, 4]

		probs = F.log_softmax(logits, dim=-1)
		return probs

	def predict(self, x: torch.Tensor, mask: torch.Tensor=None, *args):
		x = x.unsqueeze(1) 					# [batch_size, num_words, model_size] -> e.g. [12, 100, 300] -> [batch_size, 1, num_words, model_size]

		x = self.conv(x)					# [batch_size, 1, num_words, model_size] -> [batch_size, num_filters, num_words - padding, 1] e.g. [12, 300, 96, 1]
		# x = self.pooling(x)					# [batch_size, num_filters, num_words - padding, 1] -> [batch_size, num_filters, 1, 1]
		x = x.squeeze().squeeze()
		x = self.linear(x)
		logits = self.output_projection(x)	# [batch_size, num_filters] -> [batch_size, classes] e.g. [12, 4]

		probs = F.log_softmax(logits, dim=-1)
		return probs


class CommentWiseConvLinearLogSoftmax(CommentWiseConvLogSoftmax):

	def __init__(self, model_size: int, kernel_size: int, stride: int, padding: str, num_filters: int, output_size: int, sentence_lenght: int, hidden_size: int, name: str = None):
		"""Projects the output of a model like the transformer encoder to a desired shape so that it is possible to perform classification.
		This is done by projecting the output of the model (e.g. size 300 per word) down to the ammount of classes.
		This uses convolutions to predict on comment level.


		Arguments:
			hidden_size {int} -- output of the transformer encoder (d_model)
			output_size {int} -- number of classes
		"""
		super(CommentWiseConvLinearLogSoftmax, self).__init__(self, model_size, kernel_size, stride, padding, num_filters, output_size, sentence_lenght, name)

		self.hidden_size = hidden_size
		self.output_projection = nn.ModuleList(
			nn.Linear(num_filters, hidden_size),
			nn.Linear(hidden_size, output_size)
		)

class CommentWiseLinearLogSoftmax(CommentWiseConvLogSoftmax):

	def __init__(self, model_size: int, output_size: int, sentence_lenght: int, name: str = None):
		"""Projects the output of a model like the transformer encoder to a desired shape so that it is possible to perform classification.
		This is done by projecting the output of the model (e.g. size 300 per word) down to the ammount of classes.
		This uses a linear operation to project a per word prediction down to a comment prediction
		
		Arguments:
			hidden_size {int} -- output of the transformer encoder (d_model)
			output_size {int} -- number of classes
		"""
		super(CommentWiseLinearLogSoftmax, self).__init__()
		self.output_size = output_size
		self.output_projection = nn.Linear(model_size, output_size)
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
		probs = F.log_softmax(logits, dim=-1)

		return probs