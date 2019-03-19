from typing import Tuple, List, Dict, Optional, Union
import torch.nn as nn
import numpy as np
import logging
import torch
import torchtext
from tqdm import tqdm
import math
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix
import matplotlib.pyplot as plt

from data.data_loader import Dataset
from trainer.utils import *
from trainer.train_logger import TrainLogger

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

	def _compute_validation_losses(self) -> float:
		self.valid_iterator.init_epoch()
		losses = []
		for valid_batch in self.valid_iterator:
			x, y = valid_batch.comments, valid_batch.general_sentiments
			loss = self.get_loss(x, None, y)
			losses.append(loss.item())
		return np.array(losses).mean()

	def evaluate(self, iterator: torchtext.data.Iterator, show_c_matrix: bool=False, show_progress: bool=False,
				 progress_label: str="Evaluation", f1_strategy: str='micro') -> Tuple[float, float, float, np.array]:
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
				losses.append(loss.item())

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
				del y
				del loss

			avg_loss = np.array(losses).mean()
			accuracy = float(true_pos) / float(total)

			# calculate f1 score based on predictions and targets
			f_scores, p_scores, r_scores, s_scores = self.calculate_multiheaded_scores(
				predictions.data, targets, f1_strategy)
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

		self.logger.debug('Evaluation finished. Avg loss: {} - Avg: f1 {} - c_matrices: {}'.format(
			avg_loss, np.mean(f_scores), c_matrices))
		return (avg_loss, f_scores, accuracy, c_matrices)

	def evaluate_and_log_train(self, iteration: int, show_progress: bool=False, show_c_matrix=False) -> Tuple[float, float, float, float]:
		mean_train_loss = self._get_mean_loss(self.train_loss_history, iteration)
		self.train_logger.log_scalar(
			None, mean_train_loss, 'loss', 'train/mean', iteration)

		# perform validation loss
		self.logger.debug('Start Evaluation')

		try:
			mean_valid_loss, f_scores, accuracy, c_matrices = self.evaluate(self.valid_iterator, show_c_matrix=show_c_matrix,
																			 show_progress=show_progress)
		finally:
			self.change_train_mode(True)

		self.logger.debug('Evaluation Complete')
		# log results
		# report scores for each aspect
		mean_valid_f1 = np.mean(f_scores)
		names = self.model.names
		for score, name in zip(f_scores, names):
			self.logger.info(f'Aspect {name} with f1 score: {score}.')
			self.train_logger.log_scalar(None, score, 'f1', 'valid/' + name, iteration)

		self.train_logger.log_scalar(
			self.val_loss_history, mean_valid_loss, 'loss', 'valid', iteration)
		self.train_logger.log_scalar(
			self.val_acc_history, mean_valid_f1, 'f1', 'valid', iteration)
		self.dataset.baselines['current'] = accuracy
		self.train_logger.log_scalars(
			self.dataset.baselines, 'valid/accuracy', iteration)
		# log combined scalars
		self.train_logger.log_scalars({
			'train': mean_train_loss,
			'validation': mean_valid_loss
		}, 'loss', iteration)

		self.train_logger.log_confusion_matrices(c_matrices, 'valid', iteration)

		return (mean_train_loss, mean_valid_loss, mean_valid_f1, accuracy)

	def calculate_multiheaded_scores(self, prediction: torch.Tensor, targets: torch.Tensor, f1_strategy: str='micro') -> Tuple[List[float], List[float], List[float], List[float]]:
		predictions = torch.t(prediction)
		targets = torch.t(targets)

		f_scores: List[float] = []
		p_scores: List[float] = []
		r_scores: List[float] = []
		s_scores: List[float] = []

		for i in range(20):
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
				f_beta = self.calculate_f1(y_true, y_pred)
				precision = 0
				recall = 0
				support = 0

			except Exception as err:
				self.logger.exception('Could not compute f1 score for input with size {} and target size {}'.format(prediction.size(),
																								  targets.size()))
			f_scores.append(f_beta)
			p_scores.append(precision)
			r_scores.append(recall)
			s_scores.append(support)

		return f_scores, p_scores, r_scores, s_scores

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
		for i in range(4):
			metrics = self.calculate_aspect_binary_classification_result(
				target, prediction, i)
			f1 = self.calculate_binary_aspect_f1(metrics)
			if math.isnan(f1):
				f1 = 0.0
			s_f1 += f1
		return s_f1 / 4

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
			tr_loss, tr_f1, tr_accuracy, tr_c_matrices = self.evaluate(self.train_iterator, show_progress=verbose,
																   progress_label="Evaluating TRAIN")
		finally:
			self.change_train_mode(True)

		tr_f1 = np.mean(tr_f1)
		if verbose:
			self.pre_training.info('TRAIN loss:\t{}'.format(tr_loss))
			self.pre_training.info('TRAIN f1-s:\t{}'.format(tr_f1))
			self.pre_training.info('TRAIN accuracy:\t{}'.format(tr_accuracy))
		else:
			self.logger.info('TRAIN loss:\t{}'.format(tr_loss))
			self.logger.info('TRAIN f1-s:\t{}'.format(tr_f1))
			self.logger.info('TRAIN accuracy:\t{}'.format(tr_accuracy))

		self.train_logger.log_scalar(None, tr_loss, 'final', 'train/loss', 0)
		self.train_logger.log_scalar(None, tr_f1, 'final', 'train/f1', 0)

		if tr_c_matrices is not None:
			from misc.visualizer import plot_confusion_matrix
			fig = plot_confusion_matrix(tr_c_matrices, self.dataset.class_labels)
			plt.show()

		self.pre_training.debug('--- Valid Scores ---')

		try:
			val_loss, val_f1, val_accuracy, val_c_matrices = self.evaluate(self.valid_iterator, show_progress=verbose,
																	   progress_label="Evaluating VALIDATION",
																	   show_c_matrix=verbose)
		finally:
			self.change_train_mode(True)

		val_f1 = np.mean(val_f1)

		if verbose:
			self.pre_training.info('VALID loss:\t{}'.format(val_loss))
			self.pre_training.info('VALID f1-s:\t{}'.format(val_f1))
			self.pre_training.info('VALID accuracy:\t{}'.format(val_accuracy))
		else:
			self.logger.info('VALID loss:\t{}'.format(val_loss))
			self.logger.info('VALID f1-s:\t{}'.format(val_f1))
			self.logger.info('VALID accuracy:\t{}'.format(val_accuracy))

		self.train_logger.log_scalar(None, val_loss, 'final', 'train/loss', 0)
		self.train_logger.log_scalar(None, val_f1, 'final', 'train/f1', 0)
		if val_c_matrices is not None:
			from misc.visualizer import plot_confusion_matrix
			fig = plot_confusion_matrix(val_c_matrices, self.dataset.class_labels)
			plt.show()

		te_loss = -1
		te_f1 = -1
		te_c_matrices = np.zeros((10, 10))
		if use_test_set:
			self.test_iterator.train = False

			te_loss, te_f1, te_accuracy, te_c_matrices = self.evaluate(self.test_iterator, show_progress=verbose,
																	   progress_label="Evaluating TEST",
																	   show_c_matrix=verbose)
			te_f1 = np.mean(te_f1)
			if verbose:
				self.pre_training.info('TEST loss:\t{}'.format(te_loss))
				self.pre_training.info('TEST f1-s:\t{}'.format(te_f1))
				self.pre_training.info('TEST accuracy:\t{}'.format(te_accuracy))
			else:
				self.logger.info('TEST loss:\t{}'.format(te_loss))
				self.logger.info('TEST f1-s:\t{}'.format(te_f1))
				self.logger.info('TEST accuracy:\t{}'.format(te_accuracy))

			self.train_logger.log_scalar(None, te_loss, 'final', 'test/loss', 0)
			self.train_logger.log_scalar(None, te_f1, 'final', 'test/f1', 0)
			if te_c_matrices is not None:
				from misc.visualizer import plot_confusion_matrix
				fig = plot_confusion_matrix(te_c_matrices, self.dataset.class_labels)
				plt.show()

		return ((tr_loss, tr_f1, tr_c_matrices), (val_loss, val_f1, val_c_matrices), (te_loss, te_f1, te_c_matrices))

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
		
		self.train_logger.print_epoch_summary(epoch, iteration, mean_train_loss, mean_valid_loss, mean_valid_f1,
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