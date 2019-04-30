from trainer.train_evaluator import TrainEvaluator
from typing import Tuple, List, Dict, Optional, Union
import torch.nn as nn
import numpy as np
import logging
import torch
import torchtext
import math
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

from data.data_loader import Dataset
from trainer.utils import *
from trainer.train_logger import TrainLogger
from trainer.utils import ITERATOR_TEST, ITERATOR_TRAIN, ITERATOR_VALIDATION
from misc.utils import isnotebook

if isnotebook():
	from tqdm.autonotebook import tqdm
else:
	from tqdm import tqdm
	
class TrainEvaluatorCoNLL(TrainEvaluator):

	def __init__(self, *args):
		super(TrainEvaluatorCoNLL, self).__init__(*args)


	def evaluate(self, iterator: torchtext.data.Iterator, show_c_matrix: bool=True, show_progress: bool=False,
				 progress_label: str="Evaluation", f1_strategy: str='micro', iterator_name: str = 'unknwn', iteration:int=-1) -> Tuple[float, float, float, np.array, float, Tuple[int, int, int]]:
		self.logger.debug('Start evaluation at evaluation epoch of {}. Evaluate {} samples'.format(
			iterator.epoch, len(iterator)))

		# set model into evaluation mode to disable dropout
		self.change_train_mode(False)

		with torch.no_grad():

			iterator.init_epoch()

			# use batch size of 1 for evaluation
			if show_c_matrix:
				prev_batch_size = iterator.batch_size
				iterator.batch_size = 1
				bs = 1
			else:
				bs = iterator.batch_size

			losses = []
			f1_scores = []
			predictions: torch.Tensor = None
			targets: torch.Tensor = None
			c_matrices: List[np.array] = []

			if show_progress:
				iterator = tqdm(iterator, desc=progress_label, leave=False)
			true_pos = 0
			total = 0
			e_iteration = 0

			fp_sum = 0
			fn_sum = 0
			tp_sum = 0
			for batch in iterator:
				# self.logger.debug(f'Starting evaluation @{e_iteration}')
				e_iteration += 1
				x, y = batch.comments, batch.aspect_sentiments
				source_mask = None

				loss = self.get_loss(x, source_mask, y)

				# divide by batch size so that we can compare losses regardless of batch size (a higher batch size will produce a nummerically higher loss than a batch size of 1)
				losses.append(loss.item() / bs)

				# [batch_size, num_words] in the collnl2003 task num labels
				# will contain the
				# predicted class for the label
				# self.logger.debug(f'Predicting samples with size {x.size()}.')
				prediction = self.model.predict(x, source_mask)

				# calculate f1 score based on predictions and targets
				f1_macro_scores, tp, fn, fp = self.calculate_multiheaded_scores(
				iterator_name, prediction.data, y, f1_strategy, iteration=iteration, epoch=self.train_iterator.epoch)

				tp_sum += tp
				fn_sum += fn
				fp_sum += fp

				# get true positives
				# self.logger.debug('Prediction finished.  Calculating scores')
				true_pos += ((y == prediction).sum()).item()
				total += y.shape[0] * y.shape[1]

				if show_c_matrix:
					# self.logger.debug('Calculating c_matrices')
					if len(y.shape) > 1 and len(prediction.shape) > 1 and y.shape != (1, 1) and prediction.shape != (1, 1):
						y_single = y.squeeze().cpu()
						y_hat_single = prediction.squeeze().cpu()
					else:
						y_single = y.cpu()
						y_hat_single = prediction.cpu()
					c_matrices.append(confusion_matrix(
						y_single, y_hat_single, labels=range(self.num_labels)))

				# self.logger.debug(f'Evaluation iteration finished with f1 of
				# {batch_f1}.')
				# self.logger.debug('Clearing up memory')
				del batch
				del prediction
				del x
				del y
				del loss
				torch.cuda.empty_cache()


			avg_loss = np.array(losses).mean()
			accuracy = float(true_pos) / float(total)

			
			if show_c_matrix:
				self.logger.debug(f'Resetting batch size to {prev_batch_size}.')
				iterator.batch_size = prev_batch_size

			if show_c_matrix:
				self.logger.debug('Calculating confusion matrix sum')
				c_matrices = np.array(c_matrices)
				# sum element wise to get total count
				c_matrices = c_matrices.sum(axis=0)
			else:
				c_matrices = None

		# reset model into training mode
		self.change_train_mode(True)

		del losses
		torch.cuda.empty_cache()


		# calculate micro f1 score
		if tp_sum == 0:
			f1_micro = 0.0
		else:
			f1_micro = (2 * tp_sum) / (2 * tp_sum + fn_sum + fp_sum)

		self.logger.debug('Evaluation finished. Avg loss: {} - Macro F1 {} - Micro F1 {} - c_matrices: {}'.format(
			avg_loss, np.mean(f1_macro_scores), f1_micro, c_matrices))
		return (avg_loss, f1_macro_scores, accuracy, c_matrices, f1_micro, (tp_sum, fn_sum, fp_sum))


	def calculate_multiheaded_scores(self, iterator_name: str, prediction: torch.Tensor, targets: torch.Tensor, f1_strategy: str='micro', iteration: int=0, epoch: int=0) -> Tuple[List[float], int, int, int]:
		
		# for macro score
		f_scores_macro: List[float] = []

		# for micro score
		tp = 0
		fn = 0
		fp = 0

		if prediction.is_cuda:
			prediction = prediction.cpu()

		if targets.is_cuda:
			targets = targets.cpu()

		for i in range(prediction.shape[0]):
			try:
				y = prediction[i]
				y_t = targets[i]
				

				tp += (y == y_t).sum().item()
				precision, recall, f1_mean, _ = precision_recall_fscore_support(y_t, y, labels=range(10), average='micro')
				if recall != 0:
					fn += (tp-recall*tp)/recall
				if precision != 0:
					fp += (tp-(precision*tp))/precision
				f_scores_macro.append(f1_mean)

				# beta = 1.0 means f1 score
				# precision, recall, f_beta, support = precision_recall_fscore_support(y_true, y_pred, beta=1.0,
				#																	 average=f1_strategy)
				#f1_mean, cls_f1_scores, metrics = self.calculate_f1(y_true, y_pred)
				#self.train_logger.log_aspect_metrics(i, f1_mean, cls_f1_scores, metrics, iterator_name, iteration, epoch)
				precision = 0
				recall = 0
				support = 0

			except Exception as err:
				self.logger.exception('Could not compute f1 score for input with size {} and target size {}'.format(prediction.size(),
																								  targets.size()))
			# this is the macro score. From the scikit learn documentation:
			# Calculate metrics for each label, and find their unweighted mean. This does not take label imbalance into account.

			# this is for the calculation of the micro f1 score. It calculates the f1 score by summing up all tps, fps...
			# From the documentation:
			# Calculate metrics globally by counting the total true positives, false negatives and false positives.
			# However, we exclude the n/a labels which are at position 0
			# tp += sum([m['tp'] for m in metrics[1:]])
			# fn += sum([m['fn'] for m in metrics[1:]])
			# fp += sum([m['fp'] for m in metrics[1:]])
			
		return f_scores_macro, tp, fn, fp

	def calculate_f1(self, target, prediction):
		macro_f1 = 0.0
		f1_scores = []

		# target and prediction are already transposed
		eval_entries = self.get_tensor_eval_entries(target, prediction)
		tp, fp, fn, _ = self.calculate_metrics(eval_entries)

		metrics = {'tp': tp, 'fp': fp, 'fn': fn}
		empty_metric = {'tp': 0, 'fp': 0, 'fn': 0}
		micro_f1 = self.calculate_binary_aspect_f1(metrics)
		return (micro_f1, [micro_f1, micro_f1, micro_f1, micro_f1], [empty_metric, metrics, empty_metric, empty_metric])
