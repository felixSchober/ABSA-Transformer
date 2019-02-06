import logging
from typing import Tuple, List, Dict, Optional, Union, Iterable
from data.conll import ExampleList
from torch import Tensor
import torch.nn as nn

from tqdm.autonotebook import tqdm
from torchtext import data

import itertools
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix


ExampleBatch = Tuple[Tensor, Tensor, List[str], List[str], data.ReversibleField]
ExampleIterator = Iterable[ExampleBatch]

logger = logging.getLogger('prediction')


def print_samples(samples: ExampleList) -> None:
	for sample in samples:
		for word, label in sample:
			print('{} - {}'.format(word, label))
		print('\n#######################\n')

def iterate_with_sample_data(data_iterator: data.Iterator, num_samples:int=5) -> ExampleIterator:
	assert num_samples > 0
	
	data_iterator.batch_size = 1
	data_iterator.shuffle = True
	data_iterator.init_epoch()

	for i, batch in enumerate(data_iterator):
		if i > num_samples:
			break
		x = batch.inputs_word
		y = batch.labels

		reversed_input = batch.dataset.fields['inputs_word'].reverse(x)
		reversed_label = batch.dataset.fields['labels'].reverse(y)

		yield (x, y, reversed_input, reversed_label, batch.dataset.fields['labels'])

def predict_samples(model: nn.Module, data_iterator: data.Iterator, num_samples: int=5):
	iterator = iterate_with_sample_data(data_iterator, num_samples)
	for x, y, sample_text, sample_label, label_reverser in iterator:
		prediction = model.predict(x, None)
		matches = count_matches(y, prediction)
		# try to reverse prediction (might not work if model is way off)
		try:
			prediction = label_reverser.reverse(prediction)
		except IndexError as err:
			prediction = ['?']
		except Exception as err:
			logger.exception('Could not reverse prediction. Unexpected error')
		yield(sample_text, sample_label, prediction, matches)

def count_matches(y, y_hat) -> int:
	return ((y == y_hat).sum()).item()

def predict_some_examples(model: nn.Module, iterator: ExampleIterator, num_samples: int=5):
	result = []
	for x, y, y_hat, num_matches in predict_samples(model, iterator, num_samples):
		line = []
		line.append(" ".join(x))
		line.append(" ".join(y))
		line.append(" ".join(y_hat))
		line.append(num_matches)
		result.append(line)
	return result

def predict_some_examples_to_df(model: nn.Module, iterator: ExampleIterator, num_samples: int=5):
	import pandas as pd
	import numpy as np


	predictions = predict_some_examples(model, iterator, num_samples)
	return pd.DataFrame(predictions, columns=['Sentence', 'Targets', 'Predictions', '# Matches'])

def plot_confusion_matrix(c_matrix,
							classes,
							normalize=False,
							title='Confusion Matrix',
							color_map=plt.cm.Blues):
	fig = plt.figure()

	if normalize:
		c_matrix = c_matrix.astype('float') / c_matrix.sum(axis=1)[:, np.newaxis]

	plt.imshow(c_matrix, interpolation='nearest', cmap=color_map, figure=fig)
	plt.title(title, figure=fig)
	plt.colorbar()
	tick_marks = np.arange(len(classes))
	plt.xticks(tick_marks, classes, rotation=45, figure=fig)
	plt.yticks(tick_marks, classes, figure=fig)
	fmt = '.2f' if normalize else 'd'
	thresh = c_matrix.max() / 2.
	for i, j in itertools.product(range(c_matrix.shape[0]), range(c_matrix.shape[1])):
		plt.text(j, i, format(c_matrix[i, j], fmt),
				 horizontalalignment="center",
				 figure=fig,
				 color="white" if c_matrix[i, j] > thresh else "black")

	plt.ylabel('True label', figure=fig)
	plt.xlabel('Predicted label', figure=fig)
	plt.tight_layout()
	return fig