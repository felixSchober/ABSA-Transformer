import warnings
warnings.filterwarnings('ignore')  # see
								   # https://stackoverflow.com/questions/43162506/undefinedmetricwarning-f-score-is-ill-defined-and-being-set-to-0-0-in-labels-wi
import os
import logging
import time
from typing import Tuple, List, Dict, Optional, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchtext
from torch.autograd import *
from tqdm.autonotebook import tqdm
#from tqdm import tqdm

from trainer.utils import *
from trainer.train_logger import TrainLogger
from trainer.train_evaluator import TrainEvaluator
from trainer.early_stopping import EarlyStopping

from misc.utils import *
from misc.run_configuration import RunConfiguration
from data.data_loader import Dataset

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
	hyperparameters : RunConfiguration
	train_iterator : torchtext.data.Iterator
	valid_iterator : torchtext.data.Iterator
	test_iterator : torchtext.data.Iterator
	dataset : Dataset
	experiment_name : str
	experiment_number : int
	early_stopping : EarlyStopping
	checkpoint_dir : str
	log_imgage_dir : str
	seed : int
	enable_tensorboard : bool
	log_every_xth_iteration : int
	logger : logging.Logger
	logger_prediction : logging.Logger

	num_labels : int
	iterations_per_epoch_train : int
	batch_size : int
	best_model_checkpoint : ModelCheckpoint
	model_in_train: bool

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

		self._set_loggers(verbose)
		
		self.model = model
		self.loss = loss
		self.optimizer = optimizer
		self.hyperparameters = hyperparameters
		self.dataset = dataset

		self.progress_bar = None

		self.train_iterator = dataset.train_iter
		self.valid_iterator = dataset.valid_iter
		self.test_iterator = dataset.test_iter
		
		self.num_epochs = hyperparameters.num_epochs
		self.iterations_per_epoch_train = len(self.train_iterator)
		self.batch_size = self.train_iterator.batch_size

		self.checkpoint_dir = os.path.join(os.getcwd(), 'logs', experiment_name, 'checkpoints')
		self.log_imgage_dir = os.path.join(os.getcwd(), 'logs', experiment_name, 'images')
		self.seed = hyperparameters.seed
		self.log_every_xth_iteration = hyperparameters.log_every_xth_iteration

		self.pre_training.info('Classes: {}'.format(self.dataset.class_labels))
		
		self.train_logger = TrainLogger(
			experiment_name,
			self.num_epochs,
			self.pre_training,
			self.dataset.dummy_input,
			enable_tensorboard,
			self.model,
			verbose,
			hyperparameters,
			dataset,
			self.log_imgage_dir)		

		self.evaluator = TrainEvaluator(
			self.model,
			self.loss,
			self.iterations_per_epoch_train,
			self.log_every_xth_iteration,
			(dataset.train_iter, dataset.valid_iter, dataset.test_iter),
			self.train_logger,
			self.pre_training,
			dataset)

		self.early_stopping = EarlyStopping(self.optimizer, self.model, hyperparameters, self.evaluator, self.checkpoint_dir)

		self.train_logger.log_hyperparameters(self)
		self.train_logger.log_hyperparameters(self.train_logger, 'Logger', log_hp=False)
		self.train_logger.log_hyperparameters(self.evaluator, 'Evaluator', log_hp=False)


	def _set_loggers(self, verbose: bool) -> None:
		# this logger will not print to the console.  Only to the file.
		self.logger = logging.getLogger(__name__)

		# this logger will both print to the console as well as the file
		self.logger_prediction = logging.getLogger('prediction')

		if verbose:
			self.pre_training = logging.getLogger('pre_training')
		else:
			self.pre_training = logging.getLogger('pre_training_silent')	


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
		self.optimizer.zero_grad()

		# Compute loss and gradient
		loss = self.evaluator.get_loss(input, source_mask, target)

		# preform training step
		loss.backward()
		self.optimizer.step(loss)

		return loss.data	

	def load_model(self, file_name=None, custom_path=None):

		if custom_path is None:
			cp_path = self.checkpoint_dir
		else:
			cp_path = custom_path

		self.pre_training.info('Try to load model at ' + cp_path)

		if file_name is None:
			# search for checkpoint
			directory = os.fsencode(cp_path)
			for file in os.listdir(directory):
				filename = os.fsdecode(file)
				if filename.endswith('.data'):
					file_name = filename
					break
		
		if file_name is None:
			self.pre_training.error(f'Could not find checkpoint file at path {cp_path}')
			return

		path = os.path.join(cp_path, file_name)
		if os.path.isfile(path):
			self.pre_training.info(f'Load checkpoint at {path}')
			if not torch.cuda.is_available():
				checkpoint = torch.load(path, map_location='cpu')
			else:
				checkpoint = torch.load(path)

			self.evaluator.epoch = checkpoint['epoch']
			self.model.load_state_dict(checkpoint['state_dict'])
			self.optimizer.optimizer.load_state_dict(checkpoint['optimizer'])
			self.evaluator.best_f1 = checkpoint['f1']
			self.best_model_checkpoint = checkpoint
			self.pre_training.info(f'Loaded model at epoch {self.evaluator.epoch} with reported f1 of {self.evaluator.best_f1}')
			if checkpoint['hp']:
				c_hp = checkpoint['hp']
				self.pre_training.info(f'Model should be used with following hyper parameters: \n{c_hp}')

				# check if compatible with model params
				if not self.hyperparameters.run_equals(c_hp):
					self.pre_training.warn(f'Checkpoint might be incompatible with model. See parameters for checkpoint model above. Current Hyperparameters are\n{self.hyperparameters}')
				else:
					self.pre_training.info('Hyperparameters are compatible!')
			# move optimizer back to cuda 
			# see https://github.com/pytorch/pytorch/issues/2830
			if torch.cuda.is_available():
				for state in self.optimizer.optimizer.state.values():
					for k, v in state.items():
						if isinstance(v, torch.Tensor):
							state[k] = v.cuda()

		else:
			self.pre_training.error(f'Could find checkpoint at path {path}.')
		return self.model, self.optimizer, self.evaluator.epoch
	
	def set_cuda(self, use_cuda: bool=False):
		if use_cuda and torch.cuda.is_available():
			self.model = self.model.cuda()
			self.pre_training.debug('train with cuda support')
		else:
			self.pre_training.debug('train without cuda support')

	def train(self, use_cuda: bool=False, perform_evaluation: bool=True) -> TrainResult:

		self.set_cuda(use_cuda)
		self.evaluator.change_train_mode(True)
		set_seeds(self.seed)
		continue_training = True
		iterations_per_epoch = self.iterations_per_epoch_train

		self.pre_training.info('{} Iterations per epoch with batch size of {}'.format(self.iterations_per_epoch_train, self.batch_size))
		self.pre_training.info(f'Total iterations: {self.iterations_per_epoch_train * self.num_epochs}')
		self.pre_training.info('START training.')

		iteration = 0
		epoch_duration = 0
		train_duration = 0
		total_time_elapsed = 0
		train_start = time.time()
		with tqdm(total=iterations_per_epoch, leave=True, position=1) as progress_bar:
			self.train_logger.progress_bar = progress_bar
			for epoch in range(self.num_epochs):
				progress_bar.n = 0
				progress_bar.set_description(f'Epoch {epoch + 1}')

				epoch_start = time.time()

				self.logger.debug('START Epoch {}. Current Iteration {}'.format(epoch, iteration))

				# Set up the batch generator for a new epoch
				self.train_iterator.init_epoch()
				self.evaluator.epoch = epoch

				# loop iterations
				for batch in self.train_iterator:  # batch is torchtext.data.batch.Batch

					if not continue_training:
						self.logger.info('continue_training is false -> Stop training')
						break

					iteration += 1

					# self.logger.debug('Iteration ' + str(iteration))
					# Sets the module in training mode
					self.model.train()

					x, _, padding, y = batch.comments, batch.general_sentiments, batch.padding, batch.aspect_sentiments
					source_mask = create_padding_masks(padding, 1)

					train_loss = self._step(x, y, source_mask)
					self.train_logger.log_scalar(self.evaluator.train_loss_history, train_loss.item(), 'loss', 'train', iteration)
					self.train_logger.log_scalar(None, self.optimizer.rate(), 'lr', 'general', iteration)

					del train_loss
					del x
					del y
					del padding
					del source_mask

					torch.cuda.empty_cache()

					if self.log_every_xth_iteration > 0 and iteration % self.log_every_xth_iteration == 0 and iteration > 1:
						try:
							isBestResult = self.evaluator.perform_iteration_evaluation(iteration, epoch_duration, time.time() - train_start,
															train_duration)
							if isBestResult:
								self.early_stopping.reset_early_stopping(iteration, self.evaluator.best_f1)

						except Exception as err:
							self.logger.exception("Could not complete iteration evaluation")

						# ############# REMOVE ##############
						# continue_training = False
						# break
						
					progress_bar.update(1)
					progress_bar.refresh()
				# ----------- End of epoch loop -----------

				self.logger.info('End of Epoch {}'.format(self.evaluator.epoch))

				# at the end of each epoch, check the accuracies
				mean_valid_f1 = -1
				try:
					mean_train_loss, mean_valid_loss, mean_valid_f1, mean_valid_accuracy = self.evaluator.evaluate_and_log_train(iteration, show_progress=False)
					epoch_duration = time.time() - epoch_start
					self.train_logger.print_epoch_summary(epoch, iteration, mean_train_loss, mean_valid_loss, mean_valid_f1,
											mean_valid_accuracy, epoch_duration, time.time() - train_start, train_duration, self.evaluator.best_loss, self.evaluator.best_f1)

					if mean_valid_loss < self.evaluator.best_loss:
						self.evaluator.best_loss = mean_train_loss
				except Exception as err:
					self.logger.exception("Could not complete end of epoch {} evaluation")

				should_stop = self.early_stopping(mean_valid_f1, mean_valid_accuracy, iteration)
				if should_stop or not continue_training:
						continue_training = False
						break

				train_duration = self.train_logger.calculate_train_duration(self.num_epochs, epoch, time.time() - train_start, epoch_duration)

		self.logger.info('STOP training.')

		# At the end of training swap the best params into the model
		# Restore best model
		try:
			self.early_stopping.restore_best_model()
		except Exception as err:
			self.logger.exception("Could not restore best model")

		self._checkpoint_cleanup()

		self.logger.debug('Exit training')

		if perform_evaluation:
			try:
				(train_results, validation_results, test_results) = self.evaluator.perform_final_evaluation()
			except Exception as err:
				self.logger.exception("Could not perform evaluation at the end of the training.")
				train_results = (0, 0, np.zeros((12, 12)))
				validation_results = (0, 0, np.zeros((12, 12)))
				test_results = (0, 0, np.zeros((12, 12)))
		else:
			train_results = (0, 0, np.zeros((12, 12)))
			validation_results = (0, 0, np.zeros((12, 12)))
			test_results = (0, 0, np.zeros((12, 12)))

		self.train_logger.close_tb_writer()

		return {
			'model': self.model,
			'result_train': train_results,
			'result_valid': validation_results,
			'result_test': test_results
		}

	

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
				if self.evaluator.best_f1 > f1:
					# delete file
					self.logger.info(f'Deleting checkpoint file {filename} at path {checkpoint_path} with f1 of {f1}.')
					try:
						os.remove(checkpoint_path)
					except Exception as err:
						self.logger.exception(f'Could not delete checkpoint file {filename} at path {checkpoint_path}.')
	
	def get_best_loss(self):
		if self.evaluator.best_loss:
			return self.evaluator.best_loss
		return 100000

	def get_best_f1(self):
		if self.evaluator.best_f1:
			return self.evaluator.best_f1
		return 0.0

	def perform_final_evaluation(self, use_test_set: bool=True, verbose: bool=True):
		return self.evaluator.perform_final_evaluation(use_test_set, verbose)
	
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
