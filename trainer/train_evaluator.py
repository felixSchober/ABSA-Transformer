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

EvaluationResult = Tuple[float, float, np.array]

TrainResult = Dict[str, Union[nn.Module,
		EvaluationResult]]


class TrainEvaluator(object):

	def __init__(self,
				model: torch.nn.Module,
				loss: torch.nn.Module,
				iterations_per_epoch_train: int,
				log_every_xth_iteration: int,
				iterators: Tuple[torchtext.data.Iterator],
				train_logger: TrainLogger,
				pre_training: logging.Logger,
				dataset: Dataset):
		super().__init__()

		

		self.logger = logging.getLogger(__name__)
		self.model = model
		self.loss = loss
		self.iterations_per_epoch_train = iterations_per_epoch_train
		self.log_every_xth_iteration = log_every_xth_iteration
		self.train_iterator, self.valid_iterator, self.test_iterator = iterators
		self.model_in_train = None
		self.train_logger = train_logger
		self.pre_training = pre_training
		self.dataset = dataset
		self.num_labels = dataset.target_size

		self._reset()

	def change_train_mode(self, train_mode):
		# if same mode, don't change
		if self.model_in_train == train_mode:
			return

		if train_mode:
			self.model.train(mode=True)
		else:
			self.model.eval()
		self.model_in_train = train_mode

	def _reset(self) -> None:
		self.epoch = 0
		self.best_f1 = 0.0
		self.best_loss = 1000.0
		self._reset_histories()

	def _reset_histories(self) -> None:
		"""
		Resets train and val histories for the accuracy and the loss.
		"""
		self.train_loss_history = []
		self.train_acc_history = []
		self.val_acc_history = []
		self.val_loss_history = []

	def get_loss(self, input: torch.Tensor, source_mask: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
		"""Calculates loss but does not perform gradient updates

		Arguments:
			input {torch.Tensor} -- Input sample
			source_mask {torch.Tensor} -- source mask
			target {torch.Tensor} -- target tensor

		Returns:
			torch.Tensor -- loss tensor
		"""

		output = self.model(input, source_mask)
		return self.loss(output, target)

	def _get_mean_loss(self, history: List[float], iteration: int) -> float:
		iteration = iteration // self.train_iterator.batch_size
		is_end_of_epoch = iteration % self.iterations_per_epoch_train == 0 or self.log_every_xth_iteration == -1
		losses: np.array
		if is_end_of_epoch:
			losses = np.array(
				history[iteration - self.iterations_per_epoch_train:iteration])
		else:
			losses = np.array(
				history[iteration - self.log_every_xth_iteration:iteration])
		return losses.mean()

	# def _compute_validation_losses(self) -> float:
	# 	self.valid_iterator.init_epoch()
	# 	losses = []
	# 	for valid_batch in self.valid_iterator:
	# 		x, y = valid_batch.comments, valid_batch.general_sentiments
	# 		loss = self.get_loss(x, None, y)
	# 		losses.append(loss.item())
	# 	return np.array(losses).mean()

	def evaluate(self, iterator: torchtext.data.Iterator, show_c_matrix: bool=False, show_progress: bool=False,
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
			for batch in iterator:
				# self.logger.debug(f'Starting evaluation @{e_iteration}')
				e_iteration += 1
				x, y, padding = batch.comments, batch.aspect_sentiments, batch.padding
				source_mask = create_padding_masks(padding, 1)
				
				loss = self.get_loss(x, source_mask, y)

				# divide by batch size so that we can compare losses regardless of batch size (a higher batch size will produce a nummerically higher loss than a batch size of 1)
				losses.append(loss.item() / bs)

				# [batch_size, num_words] in the collnl2003 task num labels
				# will contain the
				# predicted class for the label
				# self.logger.debug(f'Predicting samples with size {x.size()}.')
				prediction = self.model.predict(x, source_mask)

				if predictions is None or targets is None:
					predictions = prediction
					targets = y
				else:
					predictions = torch.cat((predictions, prediction), 0)
					targets = torch.cat((targets, y), 0)

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
				del padding
				del y
				del loss
				torch.cuda.empty_cache()


			avg_loss = np.array(losses).mean()
			accuracy = float(true_pos) / float(total)

			# calculate f1 score based on predictions and targets
			f1_macro_scores, tp, fn, fp = self.calculate_multiheaded_scores(
				iterator_name, predictions.data, targets, f1_strategy, iteration=iteration, epoch=self.train_iterator.epoch)
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

		del predictions
		del targets
		del losses
		torch.cuda.empty_cache()


		# calculate micro f1 score
		f1_micro = (2 * tp) / (2 * tp + fn + fp)

		self.logger.debug('Evaluation finished. Avg loss: {} - Macro F1 {} - Micro F1 {} - c_matrices: {}'.format(
			avg_loss, np.mean(f1_macro_scores), f1_micro, c_matrices))
		return (avg_loss, f1_macro_scores, accuracy, c_matrices, f1_micro, (tp, fn, fp))

	def evaluate_and_log_train(self, iteration: int, show_progress: bool=False, show_c_matrix=False, f1_strategy='micro') -> Tuple[float, float, float, float]:
		mean_train_loss = self._get_mean_loss(self.train_loss_history, iteration)
		self.train_logger.log_scalar(
			None, mean_train_loss, 'loss', ITERATOR_TRAIN + '/mean', iteration)

		# perform validation loss
		self.logger.debug('Start Evaluation')

		try:
			mean_valid_loss, f_scores, accuracy, c_matrices, f1_micro, (tp, fn, fp) = self.evaluate(self.valid_iterator, show_c_matrix=show_c_matrix,
																			 show_progress=show_progress, iterator_name=ITERATOR_VALIDATION, iteration=iteration, f1_strategy=f1_strategy)
		finally:
			self.change_train_mode(True)

		self.logger.debug('Evaluation Complete')
		# log results
		# report scores for each aspect

		if f1_strategy == 'micro':
			mean_valid_f1 = f1_micro
		elif f1_strategy == 'macro':
			mean_valid_f1 = np.mean(f_scores)
		names = self.model.names

		# report head f1 scores and calculate weighted macro f1 score and unweighted macro f1 score
		for score, name in zip(f_scores, names):
			self.logger.info(f'Transformer Head {name} with f1 score: {score}.')
			self.train_logger.log_scalar(None, score, 'f1', ITERATOR_VALIDATION + '/' + name, iteration)

		self.train_logger.log_scalar(
			self.val_loss_history, mean_valid_loss, 'loss', ITERATOR_VALIDATION, iteration)
		self.train_logger.log_scalar(
			self.val_acc_history, mean_valid_f1, 'f1', ITERATOR_VALIDATION, iteration)
		self.train_logger.log_scalar(
			None, tp, 'tp', ITERATOR_VALIDATION, iteration)
		self.train_logger.log_scalar(
			None, fn, 'fn', ITERATOR_VALIDATION, iteration)
		self.train_logger.log_scalar(
			None, fp, 'fp', ITERATOR_VALIDATION, iteration)
		self.dataset.baselines['current'] = accuracy
		self.train_logger.log_scalars(
			self.dataset.baselines, ITERATOR_VALIDATION + '/accuracy', iteration)
		# log combined scalars
		self.train_logger.log_scalars({
			'train': mean_train_loss,
			'validation': mean_valid_loss
		}, 'loss', iteration)

		self.train_logger.log_confusion_matrices(c_matrices, ITERATOR_VALIDATION, iteration)

		return (mean_train_loss, mean_valid_loss, mean_valid_f1, accuracy)

	def calculate_multiheaded_scores(self, iterator_name: str, prediction: torch.Tensor, targets: torch.Tensor, f1_strategy: str='micro', iteration: int=0, epoch: int=0) -> Tuple[List[float], int, int, int]:
		predictions = torch.t(prediction)
		targets = torch.t(targets)

		# for macro score
		f_scores_macro: List[float] = []

		# for micro score
		tp = 0
		fn = 0
		fp = 0

		# iterate over all target heads and get true positives, false positives, etc for each aspect
		for i in range(len(self.dataset.target)):
			try:
				y_pred = predictions[i]
				y_true = targets[i]

				if y_pred.is_cuda:
					y_pred = y_pred.cpu()

				if y_true.is_cuda:
					y_true = y_true.cpu()

				# beta = 1.0 means f1 score
				# precision, recall, f_beta, support = precision_recall_fscore_support(y_true, y_pred, beta=1.0,
				#																	 average=f1_strategy)
				f1_mean, cls_f1_scores, metrics = self.calculate_f1(y_true, y_pred)
				self.train_logger.log_aspect_metrics(i, f1_mean, cls_f1_scores, metrics, iterator_name, iteration, epoch)
				precision = 0
				recall = 0
				support = 0

			except Exception as err:
				self.logger.exception('Could not compute f1 score for input with size {} and target size {}'.format(prediction.size(),
																								  targets.size()))
			# this is the macro score. From the scikit learn documentation:
			# Calculate metrics for each label, and find their unweighted mean. This does not take label imbalance into account.
			f_scores_macro.append(f1_mean)

			# this is for the calculation of the micro f1 score. It calculates the f1 score by summing up all tps, fps...
			# From the documentation:
			# Calculate metrics globally by counting the total true positives, false negatives and false positives.
			# However, we exclude the n/a labels which are at position 0
			tp += sum([m['tp'] for m in metrics[1:]])
			fn += sum([m['fn'] for m in metrics[1:]])
			fp += sum([m['fp'] for m in metrics[1:]])
			
		return f_scores_macro, tp, fn, fp

	def calculate_scores(self, prediction: torch.Tensor, targets: torch.Tensor, f1_strategy: str='micro') -> Tuple[List[float], List[float], List[float], List[float]]:
		p_size = prediction.size()

		if len(prediction.shape) == 1:
			labelPredictions = prediction.unsqueeze(0)
			targets = targets.unsqueeze(0)
		else:
			targets = targets.view(p_size[1], p_size[0])
			labelPredictions = prediction.view(p_size[1],
											p_size[0])  # transform prediction so that [num_labels, batch_size]

		f_scores: List[float] = []
		p_scores: List[float] = []
		r_scores: List[float] = []
		s_scores: List[float] = []

		for y_pred, y_true in zip(labelPredictions, targets):
			try:
				assert y_pred.size() == y_true.size()
				assert y_true.size()[0] > 0

				if y_pred.is_cuda:
					y_pred = y_pred.cpu()

				if y_true.is_cuda:
					y_true = y_true.cpu()

				 # beta = 1.0 means f1 score
				precision, recall, f_beta, support = precision_recall_fscore_support(y_true, y_pred, beta=1.0,
																					 average=f1_strategy)

			except Exception as err:
				self.logger.exception('Could not compute f1 score for input with size {} and target size {}'.format(prediction.size(),
																								  targets.size()))
			f_scores.append(f_beta)
			p_scores.append(precision)
			r_scores.append(recall)
			s_scores.append(support)

		return f_scores[0], p_scores, r_scores, s_scores

	def calculate_f1(self, target, prediction):
		s_f1 = 0.0
		scores = []
		metrics_list = []
		# calculate stats for each class label (e.g. n/a, pos, neg, neutr)
		for i in range(self.dataset.target_size):
			metrics = self.calculate_aspect_binary_classification_result(
				target, prediction, i)
			f1 = self.calculate_binary_aspect_f1(metrics)
			if math.isnan(f1):
				f1 = 0.0

			# for macro f1 calculation exclude n/a label
			if i > 0:
				s_f1 += f1
			metrics_list.append(metrics)
			scores.append(f1)
		return (s_f1 / (self.dataset.target_size - 1), scores, metrics_list)

	def calculate_aspect_binary_classification_result(self, target, prediction, class_label):
		mask_target = target == class_label
		mark_prediction = prediction == class_label
		c_matrix = confusion_matrix(mask_target, mark_prediction, labels=[1, 0])
		return {'tp': c_matrix[0, 0], 'fp': c_matrix[0, 1], 'fn': c_matrix[1, 0], 'tn': c_matrix[1, 1]}

	def calculate_binary_aspect_f1(self, metrics):
		return (2*metrics['tp']) / (2*metrics['tp']+metrics['fn']+metrics['fp'])

	def perform_final_evaluation(self, use_test_set: bool=True, verbose: bool=True) -> Tuple[EvaluationResult, EvaluationResult, EvaluationResult]:

		if verbose:
			self.pre_training.info('Perform final model evaluation')
			self.pre_training.debug('--- Train Scores ---')
		self.train_iterator.train = False
		self.valid_iterator.train = False

		try:
			tr_loss, tr_macro_f1, tr_accuracy, tr_c_matrices, tr_f1_micro, (tp, fn, fp) = self.evaluate(self.train_iterator, show_progress=verbose,
																   progress_label="Evaluating TRAIN", iterator_name=ITERATOR_TRAIN)
		finally:
			self.change_train_mode(True)

		if verbose:
			self.pre_training.info('TRAIN loss:\t{}'.format(tr_loss))
			self.pre_training.info('TRAIN MACRO f1-s:\t{}'.format(tr_macro_f1))
			self.pre_training.info('TRAIN MICRO f1-s:\t{}'.format(tr_f1_micro))

			self.pre_training.info('TRAIN TP:\t{}'.format(tp))
			self.pre_training.info('TRAIN FP:\t{}'.format(fn))
			self.pre_training.info('TRAIN FN:\t{}'.format(fp))

			self.pre_training.info('TRAIN accuracy:\t{}'.format(tr_accuracy))
		else:
			self.logger.info('TRAIN loss:\t{}'.format(tr_loss))
			self.logger.info('TRAIN MACRO f1-s:\t{}'.format(tr_macro_f1))
			self.logger.info('TRAIN MICRO f1-s:\t{}'.format(tr_f1_micro))

			self.logger.info('TRAIN TP:\t{}'.format(tp))
			self.logger.info('TRAIN FP:\t{}'.format(fn))
			self.logger.info('TRAIN FN:\t{}'.format(fp))			
			self.logger.info('TRAIN accuracy:\t{}'.format(tr_accuracy))

		self.train_logger.log_scalar(None, tr_loss, 'final', ITERATOR_TRAIN + '/loss', 0)
		self.train_logger.log_scalar(None, tr_f1_micro, 'final', ITERATOR_TRAIN + '/f1/micro', 0)
		self.train_logger.log_scalar(None, tr_macro_f1, 'final', ITERATOR_TRAIN + '/f1/macro', 0)


		if tr_c_matrices is not None:
			from misc.visualizer import plot_confusion_matrix
			fig = plot_confusion_matrix(tr_c_matrices, self.dataset.class_labels)
			plt.show()

		self.pre_training.debug('--- Valid Scores ---')

		try:
			val_loss, val_macro_f1, val_accuracy, val_c_matrices, val_f1_micro, (tp, fn, fp) = self.evaluate(self.valid_iterator, show_progress=verbose,
																	   progress_label="Evaluating VALIDATION",
																	   show_c_matrix=verbose, iterator_name=ITERATOR_VALIDATION)
		finally:
			self.change_train_mode(True)

		if verbose:
			self.pre_training.info('VALID loss:\t{}'.format(val_loss))
			self.pre_training.info('VALID MACRO f1-s:\t{}'.format(val_macro_f1))
			self.pre_training.info('VALID MICRO f1-s:\t{}'.format(val_f1_micro))

			self.pre_training.info('VALID TP:\t{}'.format(tp))
			self.pre_training.info('VALID FP:\t{}'.format(fn))
			self.pre_training.info('VALID FN:\t{}'.format(fp))						
			self.pre_training.info('VALID accuracy:\t{}'.format(val_accuracy))
		else:
			self.logger.info('VALID loss:\t{}'.format(val_loss))
			self.logger.info('VALID MACRO f1-s:\t{}'.format(val_macro_f1))
			self.logger.info('VALID MICRO f1-s:\t{}'.format(val_f1_micro))

			self.logger.info('VALID TP:\t{}'.format(tp))
			self.logger.info('VALID FP:\t{}'.format(fn))
			self.logger.info('VALID FN:\t{}'.format(fp))			
			self.logger.info('VALID accuracy:\t{}'.format(val_accuracy))

		self.train_logger.log_scalar(None, val_loss, 'final', ITERATOR_VALIDATION + '/loss', 0)
		self.train_logger.log_scalar(None, val_f1_micro, 'final', ITERATOR_VALIDATION + '/f1/micro', 0)
		self.train_logger.log_scalar(None, val_macro_f1, 'final', ITERATOR_VALIDATION + '/f1/macro', 0)

		if val_c_matrices is not None:
			from misc.visualizer import plot_confusion_matrix
			fig = plot_confusion_matrix(val_c_matrices, self.dataset.class_labels)
			plt.show()

		te_loss = -1
		te_f1 = -1
		te_c_matrices = np.zeros((10, 10))
		if use_test_set:
			self.test_iterator.train = False

			te_loss, te_macro_f1, te_accuracy, te_c_matrices, te_f1_micro, (tp, fn, fp) = self.evaluate(self.test_iterator, show_progress=verbose,
																	   progress_label="Evaluating TEST",
																	   show_c_matrix=verbose, iterator_name=ITERATOR_TEST)
			if verbose:
				self.pre_training.info('TEST loss:\t{}'.format(te_loss))
				self.pre_training.info('TEST MACRO f1-s:\t{}'.format(te_macro_f1))
				self.pre_training.info('TEST MICRO f1-s:\t{}'.format(te_f1_micro))

				self.pre_training.info('TEST TP:\t{}'.format(tp))
				self.pre_training.info('TEST FP:\t{}'.format(fn))
				self.pre_training.info('TEST FN:\t{}'.format(fp))							
				self.pre_training.info('TEST accuracy:\t{}'.format(te_accuracy))
			else:
				self.logger.info('TEST loss:\t{}'.format(te_loss))
				self.logger.info('TEST MACRO f1-s:\t{}'.format(te_macro_f1))
				self.logger.info('TEST MICRO f1-s:\t{}'.format(te_f1_micro))

				self.logger.info('TEST TP:\t{}'.format(tp))
				self.logger.info('TEST FP:\t{}'.format(fn))
				self.logger.info('TEST FN:\t{}'.format(fp))			
				self.logger.info('TEST accuracy:\t{}'.format(te_accuracy))

			self.train_logger.log_scalar(None, te_loss, 'final', ITERATOR_TEST + '/loss', 0)
			self.train_logger.log_scalar(None, te_f1_micro, 'final', ITERATOR_TEST + '/f1/micro', 0)			
			self.train_logger.log_scalar(None, te_macro_f1, 'final', ITERATOR_TEST + '/f1/macro', 0)

			if te_c_matrices is not None:
				from misc.visualizer import plot_confusion_matrix
				fig = plot_confusion_matrix(te_c_matrices, self.dataset.class_labels)
				plt.show()

		self.train_logger.complete_iteration(-1, -1, -1, -1,  -1, -1, -1, -1, -1, -1, -1, True)
		return ((tr_loss, tr_f1_micro, tr_c_matrices), (val_loss, val_f1_micro, val_c_matrices), (te_loss, te_f1_micro, te_c_matrices))

	def perform_iteration_evaluation(self, iteration: int, epoch_duration: float, time_elapsed: float,
										total_time: float) -> bool:
		epoch = self.epoch
		self.logger.debug('Starting evaluation in epoch {}. Current Iteration {}'.format(epoch, iteration))
		mean_train_loss, mean_valid_loss, mean_valid_f1, mean_valid_accuracy = self.evaluate_and_log_train(iteration, show_c_matrix=False)
		self.logger.debug('Evaluation completed')
		self.logger.info('Iteration {}'.format(iteration))
		self.logger.info('Mean train loss: {}'.format(mean_train_loss))
		self.logger.info('Mean validation loss {}'.format(mean_valid_loss))
		self.logger.info('Mean validation f1 score {}'.format(mean_valid_f1))
		self.logger.info('Mean validation accuracy {}'.format(mean_valid_accuracy))
		
		self.train_logger.complete_iteration(epoch, iteration, mean_train_loss, mean_valid_loss, mean_valid_f1,
								 mean_valid_accuracy, epoch_duration, time_elapsed, total_time, self.best_loss, self.best_f1)

		return self.upadate_val_scores(mean_valid_f1, mean_valid_loss)

	def upadate_val_scores(self, f1, loss) -> bool:
		best_result = False
		if f1 > self.best_f1:
			self.logger.info(f'Current f1 score of {f1} is better than last f1 score of {self.best_f1}.')
			self.best_f1 = f1
			best_result = True
		if loss < self.best_loss:
			self.best_loss = loss
		return best_result