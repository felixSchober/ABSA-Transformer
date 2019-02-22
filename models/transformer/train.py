import warnings
warnings.filterwarnings('ignore')  # see
								   # https://stackoverflow.com/questions/43162506/undefinedmetricwarning-f-score-is-ill-defined-and-being-set-to-0-0-in-labels-wi
import math
import os
import logging
import time
import datetime
import shutil
from typing import Tuple, List, Dict, Optional, Union

from sklearn.metrics import precision_recall_fscore_support, confusion_matrix
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchtext
from torch.autograd import *
from tqdm.autonotebook import tqdm
from tensorboardX import SummaryWriter
import matplotlib.pyplot as plt
from colorama import Fore, Style

from misc.visualizer import plot_confusion_matrix
from misc.utils import *
from misc.run_configuration import RunConfiguration
from misc.torchsummary import summary
from data.data_loader import Dataset

DEFAULT_CHECKPOINT_PATH = ''

ModelCheckpoint = Optional[Dict[str, Union[int,
		float,
		any]]]

EvaluationResult = Tuple[float, float, np.array]

TrainResult = Dict[str, Union[nn.Module,
		EvaluationResult]]


class Trainer(object):
	model : nn.Module
	loss : nn.Module
	optimizer : torch.optim.Optimizer
	parameters : RunConfiguration
	train_iterator : torchtext.data.Iterator
	valid_iterator : torchtext.data.Iterator
	test_iterator : torchtext.data.Iterator
	dataset : Dataset
	experiment_name : str
	experiment_number : int
	between_epochs_validation_texts : str
	early_stopping : int
	checkpoint_dir : str
	log_imgage_dir : str
	seed : int
	enable_tensorboard : bool
	log_every_xth_iteration : int
	logger : logging.Logger
	logger_prediction : logging.Logger
	tb_writer : SummaryWriter
	git_commit : str
	show_summary: bool

	num_labels : int
	epoch : int
	iterations_per_epoch_train : int
	batch_size : int
	best_f1 : int
	best_loss: int
	best_model_checkpoint : ModelCheckpoint
	early_stopping_counter : int
	train_loss_history : List[float]
	train_acc_history : List[float]
	val_acc_history : List[float]
	val_loss_history : List[float]

	def __init__(self,
				 model: nn.Module,
				 loss: nn.Module,
				 optimizer: torch.optim.Optimizer,
				 hyperparameters: RunConfiguration,
				 dataset: Dataset,
				 experiment_name: str,
				 enable_tensorboard: bool=True,
				 verbose=True):

		assert hyperparameters.log_every_xth_iteration >= -1
		assert model is not None
		assert loss is not None
		assert optimizer is not None
		assert dataset is not None

		# this logger will not print to the console.  Only to the file.
		self.logger = logging.getLogger(__name__)

		# this logger will both print to the console as well as the file
		self.logger_prediction = logging.getLogger('prediction')
		self.pre_training = logging.getLogger('pre_training')

		self.model = model
		self.loss = loss
		self.optimizer = optimizer
		self.hyperparameters = hyperparameters
		self.dataset = dataset

		self.progress_bar = None
		self.show_summary = True

		self.between_epochs_validation_texts = ''
		self.train_iterator = dataset.train_iter
		self.valid_iterator = dataset.valid_iter
		self.test_iterator = dataset.test_iter
		
		self.num_epochs = hyperparameters.num_epochs
		self.iterations_per_epoch_train = len(self.train_iterator)
		self.batch_size = self.train_iterator.batch_size
		self.experiment_name = experiment_name
		self.early_stopping = hyperparameters.early_stopping

		self.checkpoint_dir = os.path.join(os.getcwd(), 'logs', experiment_name, 'checkpoints')
		self.log_imgage_dir = os.path.join(os.getcwd(), 'logs', experiment_name, 'images')

		self.git_commit = get_current_git_commit()

		self._reset()
		self.seed = hyperparameters.seed
		self.enable_tensorboard = enable_tensorboard
		self.log_every_xth_iteration = hyperparameters.log_every_xth_iteration

		# this logger will not print to the console.  Only to the file.
		self.logger = logging.getLogger(__name__)

		# this logger will both print to the console as well as the file
		self.logger_prediction = logging.getLogger('prediction')

		if verbose:
			self.pre_training = logging.getLogger('pre_training')
		else:
			self.pre_training = logging.getLogger('pre_training_silent')

		model_summary = torch_summarize(model)
		self.pre_training.info(model_summary)

		if verbose:
			summary(self.model, input_size=(42,), dtype='long')
		self.pre_training.info(model_summary)

		self.num_labels = dataset.target_size
		self.epoch = 0
		self.pre_training.info('Classes: {}'.format(self.dataset.class_labels))
		
		if enable_tensorboard:
			self._setup_tensorboard(dataset.dummy_input, model_summary)
		self._log_hyperparameters()

		# TODO: initialize the rest of the trainings parameters
		# https://github.com/kolloldas/torchnlp/blob/master/torchnlp/common/train.py

	def _setup_tensorboard(self, dummy_input: torch.Tensor, model_summary: str) -> None:
		assert dummy_input is not None

		# construct run path
		# first level - experiment_name
		# second level - date
		# third level - num_epochs
		# fourth level - cont.  number
		run_dir = os.path.join(os.getcwd(), 'runs', self.experiment_name)
		create_dir_if_necessary(run_dir)

		now = datetime.datetime.now()
		date_identifier = now.strftime('%Y%m%d')
		run_dir = os.path.join(run_dir, date_identifier)
		create_dir_if_necessary(run_dir)

		run_dir = os.path.join(run_dir, str(self.num_epochs) + 'EP')
		create_dir_if_necessary(run_dir)

		self.experiment_number = sum(os.path.isdir(os.path.join(run_dir, i)) for i in os.listdir(run_dir))

		run_dir = os.path.join(run_dir, str(self.experiment_number))
		create_dir_if_necessary(run_dir)

		self.pre_training.info(f'Tensorboard enabled. Run will be located at /runs/{self.experiment_name}/{date_identifier}/{self.num_epochs}/{self.experiment_number}/. Full path is {run_dir}')

		# logdir = os.path.join(os.getcwd(), 'logs', experiment_name,
		# 'checkpoints')
		self.tb_writer = SummaryWriter(log_dir=run_dir, comment=self.experiment_name)

		# for now until add graph works (needs pytorch version >= 0.4) add the
		# model description as text
		self.tb_writer.add_text('model', model_summary, 0)
		self._log_text(self.git_commit, 'git')
		try:
			self.tb_writer.add_graph(self.model, dummy_input, verbose=False)
		except Exception as err:
			self.logger.exception('Could not generate graph')
		self.logger.debug('Graph Saved')

	def _reset_histories(self) -> None:
		"""
		Resets train and val histories for the accuracy and the loss.
		"""
		self.train_loss_history = []
		self.train_acc_history = []
		self.val_acc_history = []
		self.val_loss_history = []

	def _reset(self) -> None:
		self.epoch = 0
		self.best_f1 = 0.0
		self.best_loss = 1000.0
		self.best_model_checkpoint = None
		self.early_stopping_counter = self.early_stopping
		self._reset_histories()

	def print_epoch_summary(self, epoch: int, iteration: int, train_loss: float, valid_loss: float, valid_f1: float,
							valid_accuracy: float, epoch_duration: float, duration: float, total_time: float):

		color_modifier_loss = Fore.WHITE if valid_loss >= self.best_loss else Fore.GREEN
		color_modifier_f1 = Fore.WHITE if valid_f1 <= self.best_f1 else Fore.GREEN

		summary = '{0}\t{1}\t{2:.2f}\t\t{3}{4:.2f}\t\t{5}{6:.3f}{7}\t\t{8:.3f}\t\t{9:.2f}m - {10:.1f}m / {11:.1f}m'.format(
			epoch + 1, 
			iteration, 
			train_loss, 
			color_modifier_loss,
			valid_loss, 
			color_modifier_f1,
			valid_f1, 
			Style.RESET_ALL,
			valid_accuracy, 
			epoch_duration / 60, 
			duration / 60,
			total_time / 60)

		if self.show_summary:
			message = '# EP\t# IT\ttr loss\t\tval loss\tf1\t\tacc\t\tduration / total time'
			self.logger.info(message)
			self.progress_bar.write(message)
			self.show_summary = False

		self.progress_bar.write(summary)
		self.logger.info(summary)
		

	def _step(self, input: torch.Tensor, target: torch.Tensor, source_mask: torch.Tensor) -> torch.Tensor:
		"""Make a single gradient update. This is called by train() and should not
		be called manually.
		
		Arguments:
			input {torch.Tensor} -- input batch
			target {torch.Tensor} -- targets
		
		Returns:
			torch.Tensor -- loss tensor
		"""

		# Clears the gradients of all optimized :class:`torch.Tensor` s
		self.optimizer.optimizer.zero_grad()

		# Compute loss and gradient
		loss = self._get_loss(input, source_mask, target)

		# preform training step
		loss.backward()
		self.optimizer.step()

		return loss.data

	def _get_loss(self, input: torch.Tensor, source_mask: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
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
		is_end_of_epoch = iteration % self.iterations_per_epoch_train == 0 or self.log_every_xth_iteration == -1
		losses : np.array
		if is_end_of_epoch:
			losses = np.array(history[iteration - self.iterations_per_epoch_train:iteration])
		else:
			losses = np.array(history[iteration - self.log_every_xth_iteration:iteration])
		return losses.mean()

	def _log_scalar(self, history: List[float], scalar_value: float, scalar_type: str, scalar_name: str,
					iteration: int) -> None:
		if history is not None:
			history.append(scalar_value)

		if self.enable_tensorboard and self.tb_writer is not None:
			self.tb_writer.add_scalar('{}/{}'.format(scalar_name, scalar_type), scalar_value, iteration)

	def _log_scalars(self, scalar_values, scalar_type: str, iteration: int):
		if self.enable_tensorboard:
			self.tb_writer.add_scalars(scalar_type, scalar_values, iteration)

	def _log_confusion_matrices(self, c_matrices, figure_name, iteration):
		if c_matrices is not None and c_matrices != []:
			fig_abs = plot_confusion_matrix(c_matrices, self.dataset.class_labels, normalize=False)
			plt.savefig(os.path.join(self.log_imgage_dir, 'abs_c_{}.png'.format(iteration)))
			if self.enable_tensorboard and self.tb_writer is not None:
				self.tb_writer.add_figure('confusion_matrix/abs/{}'.format(figure_name), fig_abs, iteration)

			fig_norm = plot_confusion_matrix(c_matrices, self.dataset.class_labels)
			plt.savefig(os.path.join(self.log_imgage_dir, 'norm_c_{}.png'.format(iteration)))

			if self.enable_tensorboard and self.tb_writer is not None:
				self.tb_writer.add_figure('confusion_matrix/norm/{}'.format(figure_name), fig_norm, iteration)

	def _log_text(self, text: str, name: str):
		if self.enable_tensorboard and self.tb_writer is not None:
			self.tb_writer.add_text(name, text, 0)

	def _log_hyperparameters(self):
		varibable_output = get_class_variable_table(self, 'Trainer')
		self.logger.debug(varibable_output)
		self._log_text(varibable_output, 'parameters/trainer')

		varibable_output = get_class_variable_table(self.hyperparameters, 'Hyper Parameters')
		self.logger.debug(varibable_output)
		self._log_text(varibable_output, 'parameters/hyperparameters')

	def _compute_validation_losses(self) -> float:
		self.valid_iterator.init_epoch()
		losses = []
		for valid_batch in self.valid_iterator:
			x, y = valid_batch.comments, valid_batch.general_sentiments
			loss = self._get_loss(x, None, y)
			losses.append(loss.item())
		return np.array(losses).mean()

	def evaluate(self, iterator: torchtext.data.Iterator, show_c_matrix: bool=False, show_progress: bool=False,
				 progress_label: str="Evaluation", f1_strategy: str='micro') -> Tuple[float, float, float, np.array]:
		self.logger.debug('Start evaluation at evaluation epoch of {}. Evaluate {} samples'.format(iterator.epoch, len(iterator)))
		with torch.no_grad():

			iterator.init_epoch()

			# use batch size of 1 for evaluation
			if show_c_matrix:
				prev_batch_size = iterator.batch_size
				iterator.batch_size = 1

			losses = []
			f1_scores = []
			predictions : torch.Tensor = None
			targets : torch.Tensor = None
			c_matrices : List[np.array] = []

			if show_progress:
				iterator = tqdm(iterator, desc=progress_label, leave=False)
			true_pos = 0
			total = 0
			e_iteration = 0
			for batch in iterator:
				#self.logger.debug(f'Starting evaluation @{e_iteration}')
				e_iteration += 1
				x, y, padding = batch.comments, batch.aspect_sentiments, batch.padding
				source_mask = self.create_padding_masks(padding, 1)

				loss = self._get_loss(x, source_mask, y)
				losses.append(loss.item())

				# [batch_size, num_words] in the collnl2003 task num labels
				# will contain the
				# predicted class for the label
				#self.logger.debug(f'Predicting samples with size {x.size()}.')
				prediction = self.model.predict(x, source_mask)

				if predictions is None or targets is None:
					predictions = prediction
					targets = y
				else:
					predictions = torch.cat((predictions, prediction), 0)
					targets = torch.cat((targets, y), 0)

				# get true positives
				#self.logger.debug('Prediction finished.  Calculating scores')
				true_pos += ((y == prediction).sum()).item()
				total += y.shape[0] * y.shape[1]

				if show_c_matrix:
					#self.logger.debug('Calculating c_matrices')
					if len(y.shape) > 1 and len(prediction.shape) > 1 and y.shape != (1, 1) and prediction.shape != (1, 1):
						y_single = y.squeeze().cpu()
						y_hat_single = prediction.squeeze().cpu()
					else:
						y_single = y.cpu()
						y_hat_single = prediction.cpu()
					c_matrices.append(confusion_matrix(y_single, y_hat_single, labels=range(self.num_labels)))

				#self.logger.debug(f'Evaluation iteration finished with f1 of
				#{batch_f1}.')
				#self.logger.debug('Clearing up memory')
				del batch
				del prediction
				del x
				del y
				del loss

			avg_loss = np.array(losses).mean()
			accuracy = float(true_pos) / float(total)

			# calculate f1 score based on predictions and targets
			f_scores, p_scores, r_scores, s_scores = self.calculate_multiheaded_scores(predictions.data, targets, f1_strategy)
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

		self.logger.debug('Evaluation finished. Avg loss: {} - Avg: f1 {} - c_matrices: {}'.format(avg_loss, np.mean(f_scores), c_matrices))
		return (avg_loss, f_scores, accuracy, c_matrices)

	def _evaluate_and_log_train(self, iteration: int, show_progress: bool=False, show_c_matrix=True) -> Tuple[float, float, float, float]:
		mean_train_loss = self._get_mean_loss(self.train_loss_history, iteration)
		self._log_scalar(None, mean_train_loss, 'loss', 'train/mean', iteration)

		# perform validation loss
		self.logger.debug('Start Evaluation')
		mean_valid_loss, f_scores, accuracy, c_matrices = self.evaluate(self.valid_iterator, show_c_matrix=show_c_matrix,
																			 show_progress=show_progress)

		self.logger.debug('Evaluation Complete')
		# log results
		# report scores for each aspect
		mean_valid_f1 = np.mean(f_scores)
		names = self.model.names
		for score, name in zip(f_scores, names):
			self.logger.info(f'Aspect {name} with f1 score: {score}.')
			self._log_scalar(None, score, 'f1', 'valid/' + name, iteration)

		self._log_scalar(self.val_loss_history, mean_valid_loss, 'loss', 'valid', iteration)
		self._log_scalar(self.val_acc_history, mean_valid_f1, 'f1', 'valid', iteration)
		self.dataset.baselines['current'] = accuracy
		self._log_scalars(self.dataset.baselines, 'valid/accuracy', iteration)
		# log combined scalars
		self._log_scalars({
			'train': mean_train_loss,
			'validation': mean_valid_loss
		}, 'loss', iteration)

		self._log_confusion_matrices(c_matrices, 'valid', iteration)

		return (mean_train_loss, mean_valid_loss, mean_valid_f1, accuracy)

	def calculate_multiheaded_scores(self, prediction: torch.Tensor, targets: torch.Tensor, f1_strategy: str='micro') -> Tuple[List[float], List[float], List[float], List[float]]:
		predictions = torch.t(prediction)
		targets = torch.t(targets)

		f_scores : List[float] = []
		p_scores : List[float] = []
		r_scores : List[float] = []
		s_scores : List[float] = []

		for i in range(20):
			try:
				y_pred = predictions[i]
				y_true = targets[i]

				if y_pred.is_cuda:
					y_pred = y_pred.cpu()

				if y_true.is_cuda:
					y_true = y_true.cpu()

				# beta = 1.0 means f1 score
				#precision, recall, f_beta, support = precision_recall_fscore_support(y_true, y_pred, beta=1.0,
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

		f_scores : List[float] = []
		p_scores : List[float] = []
		r_scores : List[float] = []
		s_scores : List[float] = []

		for y_pred, y_true in zip(labelPredictions, targets):
			try:
				assert y_pred.size() == y_true.size()
				assert y_true.size()[0] > 0

				if y_pred.is_cuda:
					y_pred = y_pred.cpu()

				if y_true.is_cuda:
					y_true = y_true.cpu()

				 #beta = 1.0 means f1 score
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
			metrics = self.calculate_aspect_binary_classification_result(target, prediction, i)
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

	def set_cuda(self, use_cuda: bool=False):
		if use_cuda and torch.cuda.is_available():
			self.model = self.model.cuda()
			self.pre_training.debug('train with cuda support')
		else:
			self.pre_training.debug('train without cuda support')

	def train(self, use_cuda: bool=False, perform_evaluation: bool=True) -> TrainResult:

		self.set_cuda(use_cuda)
		set_seeds(self.seed)
		continue_training = True
		iterations_per_epoch = self.iterations_per_epoch_train

		self.pre_training.info('{} Iterations per epoch with batch size of {}'.format(self.iterations_per_epoch_train, self.batch_size))
		self.pre_training.info(f'Total iterations: {self.iterations_per_epoch_train * self.num_epochs}')
		self.pre_training.info('START training.')
		print('\n\n')

		iteration = 0
		epoch_duration = 0
		train_duration = 0
		total_time_elapsed = 0
		train_start = time.time()

		with tqdm(total=iterations_per_epoch, position=0) as progress_bar:
			self.progress_bar = progress_bar
			for epoch in range(self.num_epochs):
				progress_bar.n = 0
				progress_bar.set_description(f'Epoch {epoch + 1}')

				epoch_start = time.time()

				self.logger.debug('START Epoch {}. Current Iteration {}'.format(epoch, iteration))

				# Set up the batch generator for a new epoch
				self.train_iterator.init_epoch()
				self.epoch = epoch

				# loop iterations
				for batch in self.train_iterator:  # batch is torchtext.data.batch.Batch

					if not continue_training:
						self.logger.info('continue_training is false -> Stop training')
						break

					iteration += 1

					self.logger.debug('Iteration ' + str(iteration))
					# Sets the module in training mode
					self.model.train()

					x, _, padding, y = batch.comments, batch.general_sentiments, batch.padding, batch.aspect_sentiments
					source_mask = self.create_padding_masks(padding, 1)

					train_loss = self._step(x, y, source_mask)
					self._log_scalar(self.train_loss_history, train_loss.item(), 'loss', 'train', iteration)
					self._log_scalar(None, self.optimizer.rate(), 'lr', 'general', iteration)

					del train_loss
					del x
					del y
					del padding
					del source_mask

					torch.cuda.empty_cache()

					if self.log_every_xth_iteration > 0 and iteration % self.log_every_xth_iteration == 0 and iteration > 1:
						try:
							self._perform_iteration_evaluation(iteration, epoch_duration, time.time() - train_start,
															train_duration)
						except Exception as err:
							self.logger.exception("Could not complete iteration evaluation")
					progress_bar.update(1)
					progress_bar.refresh()
				# ----------- End of epoch loop -----------

				self.logger.info('End of Epoch {}'.format(self.epoch))

				# at the end of each epoch, check the accuracies
				mean_valid_f1 = -1
				try:
					mean_train_loss, mean_valid_loss, mean_valid_f1, mean_valid_accuracy = self._evaluate_and_log_train(iteration, show_progress=False)
					epoch_duration = time.time() - epoch_start
					self.print_epoch_summary(epoch, iteration, mean_train_loss, mean_valid_loss, mean_valid_f1,
											mean_valid_accuracy, epoch_duration, time.time() - train_start, train_duration)

					if mean_valid_loss < self.best_loss:
						self.best_loss = mean_train_loss
				except Exception as err:
					self.logger.exception("Could not complete end of epoch {} evaluation")

				# early stopping if no improvement of val_acc during the last
				# self.early_stopping epochs
				# https://link.springer.com/chapter/10.1007/978-3-642-35289-8_5
				if mean_valid_f1 > self.best_f1 or self.early_stopping <= 0:
					self.logger.info(f'Current f1 score of {mean_valid_f1} (acc of {mean_valid_accuracy} is better than last f1 score of {self.best_f1}.')
					self._reset_early_stopping(iteration, mean_valid_f1)
				else:
					should_stop = self._perform_early_stopping()
					if should_stop:
						continue_training = False
						break

				train_duration = self.calculate_train_duration(self.num_epochs, epoch, time.time() - train_start, epoch_duration)

		self.logger.info('STOP training.')

		# At the end of training swap the best params into the model
		# Restore best model
		try:
			self._restore_best_model()
		except Exception as err:
			self.logger.exception("Could not restore best model")

		self._checkpoint_cleanup()

		self.logger.debug('Exit training')

		if perform_evaluation:
			try:
				(train_results, validation_results, test_results) = self.perform_final_evaluation()
			except Exception as err:
				self.logger.exception("Could not perform evaluation at the end of the training.")
				train_results = (0, 0, np.zeros((12, 12)))
				validation_results = (0, 0, np.zeros((12, 12)))
				test_results = (0, 0, np.zeros((12, 12)))
		else:
			train_results = (0, 0, np.zeros((12, 12)))
			validation_results = (0, 0, np.zeros((12, 12)))
			test_results = (0, 0, np.zeros((12, 12)))

		self._close_tb_writer()

		return {
			'model': self.model,
			'result_train': train_results,
			'result_valid': validation_results,
			'result_test': test_results
		}

	def calculate_train_duration(self, num_epochs: int, current_epoch: int, time_elapsed: float,
								 epoch_duration: float) -> float:
		# calculate approximation of time for remaining epochs
		epochs_remaining = num_epochs - (current_epoch + 1)
		duration_for_remaining_epochs = epochs_remaining * epoch_duration

		total_estimated_duration = time_elapsed + duration_for_remaining_epochs
		return total_estimated_duration

	def create_padding_masks(self, targets: torch.Tensor, padd_class: int) -> torch.Tensor:
		input_mask = (targets != padd_class).unsqueeze(-2)
		return input_mask

	def perform_final_evaluation(self, use_test_set: bool=True, verbose: bool=True) -> Tuple[EvaluationResult, EvaluationResult, EvaluationResult]:
		
		if verbose:
			self.pre_training.info('Perform final model evaluation')
			self.pre_training.debug('--- Train Scores ---')
		self.train_iterator.train = False
		self.valid_iterator.train = False

		tr_loss, tr_f1, tr_accuracy, tr_c_matrices = self.evaluate(self.train_iterator, show_progress=verbose,
																   progress_label="Evaluating TRAIN")
		tr_f1 = np.mean(tr_f1)
		if verbose:
			self.pre_training.info('TRAIN loss:\t{}'.format(tr_loss))
			self.pre_training.info('TRAIN f1-s:\t{}'.format(tr_f1))
			self.pre_training.info('TRAIN accuracy:\t{}'.format(tr_accuracy))
		else:
			self.logger.info('TRAIN loss:\t{}'.format(tr_loss))
			self.logger.info('TRAIN f1-s:\t{}'.format(tr_f1))
			self.logger.info('TRAIN accuracy:\t{}'.format(tr_accuracy))

		self._log_scalar(None, tr_loss, 'final', 'train/loss', 0)
		self._log_scalar(None, tr_f1, 'final', 'train/f1', 0)

		if tr_c_matrices is not None:
			fig = plot_confusion_matrix(tr_c_matrices, self.dataset.class_labels)
			plt.show()

		self.pre_training.debug('--- Valid Scores ---')
		val_loss, val_f1, val_accuracy, val_c_matrices = self.evaluate(self.valid_iterator, show_progress=verbose,
																	   progress_label="Evaluating VALIDATION",
																	   show_c_matrix=verbose)
		val_f1 = np.mean(val_f1)

		if verbose:
			self.pre_training.info('VALID loss:\t{}'.format(val_loss))
			self.pre_training.info('VALID f1-s:\t{}'.format(val_f1))
			self.pre_training.info('VALID accuracy:\t{}'.format(val_accuracy))
		else:
			self.logger.info('VALID loss:\t{}'.format(val_loss))
			self.logger.info('VALID f1-s:\t{}'.format(val_f1))
			self.logger.info('VALID accuracy:\t{}'.format(val_accuracy))

		self._log_scalar(None, val_loss, 'final', 'train/loss', 0)
		self._log_scalar(None, val_f1, 'final', 'train/f1', 0)
		if val_c_matrices is not None:
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

			self._log_scalar(None, te_loss, 'final', 'test/loss', 0)
			self._log_scalar(None, te_f1, 'final', 'test/f1', 0)
			if te_c_matrices is not None:
				fig = plot_confusion_matrix(te_c_matrices, self.dataset.class_labels)
				plt.show()

		return ((tr_loss, tr_f1, tr_c_matrices), (val_loss, val_f1, val_c_matrices), (te_loss, te_f1, te_c_matrices))

	def _perform_iteration_evaluation(self, iteration: int, epoch_duration: float, time_elapsed: float,
									  total_time: float) -> None:
		self.logger.debug('Starting evaluation in epoch {}. Current Iteration {}'.format(self.epoch, iteration))
		mean_train_loss, mean_valid_loss, mean_valid_f1, mean_valid_accuracy = self._evaluate_and_log_train(iteration, show_c_matrix=False)
		self.logger.debug('Evaluation completed')
		self.logger.info('Iteration {}'.format(iteration))
		self.logger.info('Mean train loss: {}'.format(mean_train_loss))
		self.logger.info('Mean validation loss {}'.format(mean_valid_loss))
		self.logger.info('Mean validation f1 score {}'.format(mean_valid_f1))
		self.logger.info('Mean validation accuracy {}'.format(mean_valid_accuracy))
		
		self.print_epoch_summary(self.epoch, iteration, mean_train_loss, mean_valid_loss, mean_valid_f1,
								 mean_valid_accuracy, epoch_duration, time_elapsed, total_time)

		if mean_valid_f1 > self.best_f1:
			self.logger.info(f'Current f1 score of {mean_valid_f1} is better than last f1 score of {self.best_f1}.')
			self._reset_early_stopping(iteration, mean_valid_f1)
		if mean_valid_loss < self.best_loss:
			self.best_loss = mean_valid_loss

	def _reset_early_stopping(self, iteration: int, mean_valid_f1: float) -> None:
		self.logger.info('Epoch f1 score ({}) better than last f1 score ({}). Save checkpoint'.format(mean_valid_f1, self.best_f1))
		self.best_f1 = mean_valid_f1

		# Save best model
		self.best_model_checkpoint = {
			'epoch': self.epoch,
			'state_dict': self.model.state_dict(),
			'val_acc': mean_valid_f1,
			'optimizer': self.optimizer.optimizer.state_dict(),
		}
		self._save_checkpoint(iteration)

		# restore early stopping counter
		self.early_stopping_counter = self.early_stopping

	def _perform_early_stopping(self) -> bool:
		self.early_stopping_counter -= 1

		# if early_stopping_counter is 0 restore best weights and stop training
		if self.early_stopping > -1 and self.early_stopping_counter <= 0:
			self.logger.info('> Early Stopping after {} epochs of no improvements.'.format(self.early_stopping))
			self.logger.info('> Restoring params of best model with validation accuracy of: {}'.format(self.best_f1))

			# Restore best model
			self._restore_best_model()
			return True
		else:
			return False

	def _close_tb_writer(self) -> None:
		if not self.enable_tensorboard or self.tb_writer is None:
			return

		if self.tb_writer is not None:
			self.logger.debug('Try to write scalars file and close tensorboard writer')
			try:
				self.tb_writer.export_scalars_to_json(os.path.join(os.getcwd(), 'logs', self.experiment_name, "model_all_scalars.json"))
			except Exception as err:
				self.logger.exception('TensorboardX could not save scalar json values')
			finally:
				self.tb_writer.close()
				self.tb_writer = None
				self.enable_tensorboard = False

	def _restore_best_model(self) -> None:
		try:
			self.model.load_state_dict(self.best_model_checkpoint['state_dict'])
		except Exception as err:
			self.logger.exception('Could not restore best model ')

		try:
			self.optimizer.optimizer.load_state_dict(self.best_model_checkpoint['optimizer'])
		except Exception as err:
			self.logger.exception('Could not restore best model ')

		self.logger.info('Best model parameters were at \nEpoch {}\nValidation f1 score {}'
						 .format(self.best_model_checkpoint['epoch'], self.best_model_checkpoint['val_acc']))

	def _checkpoint_cleanup(self):
		path = self.checkpoint_dir
		self.logger.info('Cleaning up old checkpoints')
		directory = os.fsencode(path)
		for file in os.listdir(directory):
			filename = os.fsdecode(file)
			if filename.endswith('.data'):
				checkpoint_path = os.path.join(path, filename)
				self.logger.debug(f'Loading checkpoint file {filename} at path {checkpoint_path}')

				if not torch.cuda.is_available():
					checkpoint = torch.load(checkpoint_path, map_location='cpu')
				else:
					checkpoint = torch.load(checkpoint_path)

				if 'f1' in checkpoint:                    
					f1 = checkpoint['f1']
				else:
					f1 = 0.0

				# worse than best f1 -> delete
				if self.best_f1 > f1:
					# delete file
					self.logger.info(f'Deleting checkpoint file {filename} at path {checkpoint_path} with f1 of {f1}.')
					try:
						os.remove(checkpoint_path)
					except Exception as err:
						self.logger.exception(f'Could not delete checkpoint file {filename} at path {checkpoint_path}.')

	def load_model(self, file_name=None):
		if file_name is None:
			# search for checkpoint
			directory = os.fsencode(self.checkpoint_dir)
			for file in os.listdir(directory):
				filename = os.fsdecode(file)
				if filename.endswith('.data'):
					file_name = filename
					break
		
		if file_name is None:
			self.logger.error(f'Could not find checkpoint file at path {self.checkpoint_dir}')
			return

		path = os.path.join(self.checkpoint_dir, file_name)
		if os.path.isfile(path):
			self.pre_training.info(f'Load checkpoint at {path}')
			if not torch.cuda.is_available():
				checkpoint = torch.load(path, map_location='cpu')
			else:
				checkpoint = torch.load(path)

			self.epoch = checkpoint['epoch']
			self.model.load_state_dict(checkpoint['state_dict'])
			self.optimizer.optimizer.load_state_dict(checkpoint['optimizer'])
			self.best_f1 = checkpoint['f1']
			self.best_model_checkpoint = checkpoint
			self.pre_training.info(f'Loaded model at epoch {self.epoch} with reported f1 of {self.best_f1}')

			# move optimizer back to cuda 
			# see https://github.com/pytorch/pytorch/issues/2830
			if torch.cuda.is_available():
				for state in self.optimizer.optimizer.state.values():
					for k, v in state.items():
						if isinstance(v, torch.Tensor):
							state[k] = v.cuda()

		else:
			self.pre_training.error(f'Could find checkpoint at path {path}.')
		return self.model, self.optimizer, self.epoch

	def _save_checkpoint(self, iteration: int) -> None:
		self.logger.debug('Saving model... ' + self.checkpoint_dir)

		checkpoint = {
			'iteration': iteration,
			'state_dict': self.model.state_dict(),
			'optimizer': self.optimizer.optimizer.state_dict(),
			'epoch': self.epoch,
			'f1': self.best_f1
		}

		filename = 'checkpoint_{}.data'.format(iteration)
		try:
			torch.save(checkpoint, os.path.join(self.checkpoint_dir, filename))
			# shutil.copyfile(filename, os.path.join(self.checkpoint_dir,
			# filename))
		except Exception as err:
			self.logger.exception('Could not save model.')

	def classify_sentence(self, sentence: str) -> str:
		x = self.manual_process(sentence, self.dataset.source_reverser)
		# if self.model.is_cuda:
		x = x.cuda()
		y_hat = self.model.predict(x)
		predicted_labels = self.dataset.target_reverser.reverse(y_hat)
		return predicted_labels

	def manual_process(self, input: str, data_field: torchtext.data.field.Field) -> torch.Tensor:
		preprocessed_input = data_field.preprocess(input)

		# strip spaces
		preprocessed_input = [x.strip(' ') for x in preprocessed_input]
		preprocessed_input = [preprocessed_input]
		input_tensor = data_field.process(preprocessed_input)
		return input_tensor
