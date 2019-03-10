import logging
import os
import time
import datetime
import torch
from tensorboardX import SummaryWriter
from misc.utils import *
from misc.visualizer import plot_confusion_matrix
from misc.torchsummary import summary
from misc.run_configuration import RunConfiguration
from colorama import Fore, Style
from data.data_loader import Dataset
from typing import Tuple, List, Dict, Optional, Union
import matplotlib.pyplot as plt


class TrainLogger(object):

	def __init__(self,
				experiment_name: str,
				num_epochs: int,
				pre_training: logging.Logger,
				dummy_input: torch.Tensor,
				enable_tensorboard: bool,
				model: torch.nn.Module,
				verbose: bool,
				hp: RunConfiguration,
				dataset: Dataset,
				log_image_dir: str):
		super().__init__()

		self.experiment_name = experiment_name
		self.num_epochs = num_epochs
		self.experiment_number = None
		self.tb_writer = None
		self.pre_training = pre_training
		self.logger = logging.getLogger(__name__)
		self.enable_tensorboard = enable_tensorboard
		self.model = model
		self.verbose = verbose
		self.hyperparameters = hp
		self.progress_bar = None
		self.dataset = dataset
		self.log_imgage_dir = log_image_dir
		self.git_commit = get_current_git_commit()
		self._initialize(dummy_input)
		self.show_summary = True

	def _initialize(self, dummy_input: torch.Tensor):
		model_summary = torch_summarize(self.model)
		self.pre_training.info(model_summary)

		if self.verbose:
			summary(self.model, input_size=(
				self.hyperparameters.clip_comments_to,), dtype='long')

		if self.enable_tensorboard:
			self._setup_tensorboard(dummy_input, model_summary)

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

			self.experiment_number = sum(os.path.isdir(
				os.path.join(run_dir, i)) for i in os.listdir(run_dir))

			run_dir = os.path.join(run_dir, str(self.experiment_number))
			create_dir_if_necessary(run_dir)

			self.pre_training.info(
				f'Tensorboard enabled. Run will be located at /runs/{self.experiment_name}/{date_identifier}/{self.num_epochs}/{self.experiment_number}/. Full path is {run_dir}')

			# logdir = os.path.join(os.getcwd(), 'logs', experiment_name,
			# 'checkpoints')
			self.tb_writer = SummaryWriter(
				log_dir=run_dir, comment=self.experiment_name)

			# for now until add graph works (needs pytorch version >= 0.4) add the
			# model description as text
			self.tb_writer.add_text('model', model_summary, 0)
			self.log_text(self.git_commit, 'git')
			try:
				self.tb_writer.add_graph(
					self.model, dummy_input, verbose=False)
			except Exception as err:
				self.logger.exception('Could not generate graph')
			self.logger.debug('Graph Saved')

	def print_epoch_summary(self, epoch: int, iteration: int, train_loss: float, valid_loss: float, valid_f1: float,
							valid_accuracy: float, epoch_duration: float, duration: float, total_time: float, best_loss: float, best_f1: float):

		color_modifier_loss = Fore.WHITE if valid_loss > best_loss else Fore.GREEN
		color_modifier_f1 = Fore.WHITE if valid_f1 < best_f1 else Fore.GREEN

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

	def log_scalar(self, history: List[float], scalar_value: float, scalar_type: str, scalar_name: str,
					iteration: int) -> None:
		if history is not None:
			history.append(scalar_value)

		if self.enable_tensorboard and self.tb_writer is not None:
			self.tb_writer.add_scalar(
				'{}/{}'.format(scalar_name, scalar_type), scalar_value, iteration)

	def log_scalars(self, scalar_values, scalar_type: str, iteration: int):
		if self.enable_tensorboard:
			self.tb_writer.add_scalars(scalar_type, scalar_values, iteration)

	def log_confusion_matrices(self, c_matrices, figure_name, iteration):
		if c_matrices is not None and c_matrices != []:
			fig_abs = plot_confusion_matrix(
				c_matrices, self.dataset.class_labels, normalize=False)
			plt.savefig(os.path.join(self.log_imgage_dir,
						'abs_c_{}.png'.format(iteration)))
			if self.enable_tensorboard and self.tb_writer is not None:
				self.tb_writer.add_figure(
					'confusion_matrix/abs/{}'.format(figure_name), fig_abs, iteration)

			fig_norm = plot_confusion_matrix(c_matrices, self.dataset.class_labels)
			plt.savefig(os.path.join(self.log_imgage_dir,
						'norm_c_{}.png'.format(iteration)))

			if self.enable_tensorboard and self.tb_writer is not None:
				self.tb_writer.add_figure(
					'confusion_matrix/norm/{}'.format(figure_name), fig_norm, iteration)

	def log_text(self, text: str, name: str):
		if self.enable_tensorboard and self.tb_writer is not None:
			self.tb_writer.add_text(name, text, 0)

	def log_hyperparameters(self, cls, name='Trainer', log_hp=True):
		varibable_output = get_class_variable_table(cls, name)
		self.logger.debug(varibable_output)
		self.log_text(varibable_output, f'parameters/{name}')

		if log_hp:
			varibable_output = get_class_variable_table(
				self.hyperparameters, 'Hyper Parameters')
			self.logger.debug(varibable_output)
			self.log_text(varibable_output, 'parameters/hyperparameters')

	def calculate_train_duration(self, num_epochs: int, current_epoch: int, time_elapsed: float,
								 epoch_duration: float) -> float:
		# calculate approximation of time for remaining epochs
		epochs_remaining = num_epochs - (current_epoch + 1)
		duration_for_remaining_epochs = epochs_remaining * epoch_duration

		total_estimated_duration = time_elapsed + duration_for_remaining_epochs
		return total_estimated_duration

	def close_tb_writer(self) -> None:
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
