import os
import math
import logging
import torch
from misc.run_configuration import RunConfiguration

from trainer.train_evaluator import TrainEvaluator

class EarlyStopping(object):

	def __init__(self, optimizer: torch.nn.Module, model: torch.nn.Module, hp: RunConfiguration, evaluator: TrainEvaluator, checkpoint_dir: str):
		self.early_stopping = hp.early_stopping
		self.early_stopping_counter = hp.early_stopping
		self.checkpoint_dir = checkpoint_dir
		self.best_model_checkpoint = None

		self.evaluator = evaluator
		self.optimizer = optimizer
		self.model = model
		self.hp = hp

		self.should_stop = False

		# this logger will not print to the console.  Only to the file.
		self.logger = logging.getLogger(__name__)

	def reset_early_stopping(self, iteration: int, mean_valid_f1: float):
		self.logger.info('Epoch f1 score ({}) better than last f1 score ({}). Save checkpoint'.format(mean_valid_f1, self.evaluator.best_f1))
		self.evaluator.best_f1 = mean_valid_f1

		# Save best model
		best_model_checkpoint = {
			'iteration': iteration,
			'epoch': self.evaluator.epoch,
			'state_dict': self.model.state_dict(),
			'val_acc': mean_valid_f1,
			'optimizer': self.optimizer.optimizer.state_dict(),
			'f1': self.evaluator.best_f1,
			'hp': self.hp
		}
		self.best_model_checkpoint = best_model_checkpoint
		self._save_checkpoint(iteration)

		# restore early stopping counter
		self.early_stopping_counter = self.early_stopping

	def perform_early_stopping(self, isNan:bool=False) -> bool:
		self.early_stopping_counter -= 1

		if isNan:
			self.early_stopping_counter -= 2

		# if early_stopping_counter is 0 restore best weights and stop training
		if self.early_stopping > -1 and self.early_stopping_counter <= 0:
			self.logger.info('> Early Stopping after {} epochs of no improvements.'.format(self.early_stopping))
			self.logger.info('> Restoring params of best model with validation accuracy of: {}'.format(self.evaluator.best_f1))

			# Restore best model
			self.restore_best_model()
			return True
		else:
			return False

	def _save_checkpoint(self, iteration: int) -> None:
		self.logger.debug('Saving model... ' + self.checkpoint_dir)

		checkpoint = {
			'iteration': iteration,
			'state_dict': self.model.state_dict(),
			'optimizer': self.optimizer.optimizer.state_dict(),
			'epoch': self.evaluator.epoch,
			'f1': self.evaluator.best_f1,
			'hp': self.hp
		}

		filename = 'checkpoint_{}.data'.format(iteration)
		try:
			torch.save(checkpoint, os.path.join(self.checkpoint_dir, filename))
		except Exception as err:
			self.logger.exception('Could not save model.')

	def restore_best_model(self) -> None:
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

	def __call__(self, loss: float, f1: float, accuracy: float, iteration: int) -> bool:

		# if the loss is not a number we don't want to continue to long
		isLossNaN = math.isnan(loss)
		if isLossNaN:
			self.logger.info(f'Loss is NaN. > Reduce early stopping counter by 3')

        # early stopping if no improvement of val_acc during the last
        # https://link.springer.com/chapter/10.1007/978-3-642-35289-8_5

		# to reset early stopping we need to fulfill the following conditions:
		# 	1: Loss should be a number
		#	2: current f1 should be better than prev. best
		#	3: current loss should be better than prev. best
		#	4: early stopping is enabled
		if not isLossNaN and ((f1 > self.evaluator.best_f1 or loss < self.evaluator.best_loss) or not self.isEnabled):
			self.logger.info(f'Current f1 score of {f1} (acc of {accuracy} is better than last f1 score of {self.evaluator.best_f1}.')
			self.reset_early_stopping(iteration, f1)
		else:
			self.should_stop = self.perform_early_stopping()
            
		return self.should_stop

	@getattr
	def isEnabled(self):
		return self.early_stopping > 0
