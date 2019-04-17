import matplotlib
import copy
import logging
import time
import math
import os

from data.transfer_data_loader import TransferDataLoader

from misc.preferences import PREFERENCES
from misc.run_configuration import get_default_params, randomize_params, OutputLayerType, hyperOpt_goodParams, elmo_params, good_organic_hp_params
from misc import utils

from optimizer import get_optimizer
from criterion import NllLoss, LossCombiner

from models.transformer.encoder import TransformerEncoder
from models.jointAspectTagger import JointAspectTagger
from trainer.train import Trainer

import pprint
import torch
import pandas as pd

STATUS_FAIL = 'fail'
STATUS_OK = 'ok'

class TransferLearningExperiment(object):

	def __init__(self, task, experiment_name, experiment_description, default_hp, overwrite_hp, data_loaders, dataset_infos, runs=5):

		# make sure preferences are set
		assert data_loaders is not None
		assert len(data_loaders) == len(dataset_infos["data_root"])
		assert runs > 0

		self.task = task
		self.experiment_name = experiment_name
		self.experiment_description = experiment_description
		self.default_hp = default_hp
		self.overwrite_hp = overwrite_hp
		self.use_cuda = torch.cuda.is_available()
		self.dsls = data_loaders
		self.dataset_infos = dataset_infos
		self.runs = runs
		self.hp = None
		self.data_frame = pd.DataFrame()

		print(f'Transfer Learning Experiment {self.experiment_name} initialized. Source: {dataset_infos["data_root"][0]} -> Target {dataset_infos["data_root"][1]}')
		

	def _initialize(self):
		# make sure the seed is not set if more than one run
		if self.runs > 1:
			seed = None
		else:
			seed = 42
		self.overwrite_hp = {**self.overwrite_hp, **{'seed': seed, 'task': self.task}}
		self.hp = get_default_params(use_cuda=self.use_cuda, overwrite=self.overwrite_hp, from_default=self.default_hp)
		assert self.hp.seed == seed
		assert self.hp.task == self.task

	def load_model(self, dataset, rc, experiment_name, iteration):
		loss = LossCombiner(4, dataset.class_weights, NllLoss)

		if iteration == 0:
			self.current_transformer = TransformerEncoder(dataset.source_embedding,
											hyperparameters=rc)
		model = JointAspectTagger(self.current_transformer, rc, 4, 20, dataset.target_names)
		optimizer = get_optimizer(model, rc)
		trainer = Trainer(
							model,
							loss,
							optimizer,
							rc,
							dataset,
							experiment_name,
							enable_tensorboard=True,
							verbose=True)
		return trainer

	def load_dataset(self, rc, logger, task):
		dataset = TransferDataLoader(
			name=task,
			logger=logger,
			configuration=rc,
			source_index=self.dataset_infos['source_index'],
			target_vocab_index=self.dataset_infos['target_vocab_index'],
			data_path=self.dataset_infos['data_root'],
			train_file=self.dataset_infos['data_train'],
			valid_file=self.dataset_infos['data_validation'],
			test_file=self.dataset_infos['data_test'],
			file_format=self.dataset_infos['file_format'],
			init_token=None,
			eos_token=None
		)
		dataset_generator = dataset.load_data(self.dsls, verbose=False)
		self.current_dataset = dataset
		return dataset_generator

	def _objective(self, rc, run):
		run_time = time.time()
    
		# reset loggers
		utils.reset_loggers()
		experiment_name = utils.create_loggers(experiment_name=self.experiment_name)
		logger = logging.getLogger(__name__)
		dataset_logger = logging.getLogger('data_loader')
		
		logger.info(f'Experiment: [{run}/{self.runs}]')
		logger.info('Name: ' + self.experiment_name)
		logger.info('Actual Path Name: ' + experiment_name)
		logger.info('Description: ' + self.experiment_description)
		
		print('\n\n#########################################################################')
		print('Name: ' + self.experiment_name)
		print('Description: ' + self.experiment_description)
		print('#########################################################################\n\n')
		print(rc)

		logger.debug('Load dataset')
		results = []
		try:
			dataset_generator = self.load_dataset(rc, dataset_logger, rc.task)
		except Exception as err:
			print('Could load dataset: ' + str(err))
			logger.exception("Could not load dataset")
			return {
				'status': STATUS_FAIL,
				'eval_time': time.time() - run_time
			}
		logger.debug('dataset loaded')

		for i in dataset_generator:
			logger.debug(f'Load model [{i}/{len(self.dsls)}]')
			print(f'Load model [{i+1}/{len(self.dsls)}]')

			try:
				trainer = self.load_model(self.current_dataset, rc, experiment_name, i)
			except Exception as err:
				print('Could not load model: ' + str(err))
				logger.exception("Could not load model")
				return {
					'status': STATUS_FAIL,
					'eval_time': time.time() - run_time
				}

			logger.debug(f'model {i} loaded')
			logger.debug('Begin training')
			model = None
			try:
				result = trainer.train(use_cuda=rc.use_cuda, perform_evaluation=False)
				model = result['model']
			except Exception as err:
				print('Exception while training: ' + str(err))
				logger.exception("Could not complete iteration")
				return {
					'status': STATUS_FAIL,
					'eval_time': time.time() - run_time,
					'best_loss': trainer.get_best_loss(),
					'best_f1': trainer.get_best_f1()
				}

			if math.isnan(trainer.get_best_loss()):
				print('Loss is nan')
				return {
					'status': STATUS_FAIL,
					'eval_time': time.time() - run_time,
					'best_loss': trainer.get_best_loss(),
					'best_f1': trainer.get_best_f1()
				}

			# perform evaluation and log results
			result = None
			try:
				result = trainer.perform_final_evaluation(use_test_set=True, verbose=False)
			except Exception as err:
				logger.exception("Could not complete iteration evaluation.")
				print('Could not complete iteration evaluation: ' + str(err))
				return {
					'status': STATUS_FAIL,
					'eval_time': time.time() - run_time,
					'best_loss': trainer.get_best_loss(),
					'best_f1': trainer.get_best_f1()
				}
			print(f'VAL f1\t{trainer.get_best_f1()} - ({result[1][1]})')
			print(f'VAL loss\t{trainer.get_best_loss()}')
			results.append({
					'loss': result[1][0],
					'status': STATUS_OK,
					'eval_time': time.time() - run_time,
					'best_loss': trainer.get_best_loss(),
					'best_f1': trainer.get_best_f1(),
					'sample_iterations': trainer.get_num_samples_seen(),
					'iterations': trainer.get_num_iterations(),
					'rc': rc,
					'results': {
						'train': {
							'loss': result[0][0],
							'f1': result[0][1]
						},
						'validation': {
							'loss': result[1][0],
							'f1': result[1][1]
						},
						'test': {
							'loss': result[2][0],
							'f1': result[2][1]
						}
					}
				})
		return results

	def run(self):
		e_name = utils.create_loggers(experiment_name=self.experiment_name)
		e_path = os.path.join(os.getcwd(), 'logs', e_name)
		logger = logging.getLogger(__name__)
		logger.info('#########################################################################')
		logger.info(f'### Experiment:\t{self.experiment_name}')
		logger.info(f'### Description:\t{self.experiment_description}')
		logger.info(f'### Runs:\t\t{self.runs}')
		logger.info('#########################################################################')

		results = []
		for i in range(self.runs):
			self._initialize()
			result = self._objective(self.hp, i)[-1]
			self._print_result(result, i)

			df_row = {}
			df_row['run'] = i
			df_row['status'] = result['status']
			df_row['time'] = result['eval_time']

			if result['status'] == STATUS_OK:
				results.append(result)

				r_tr = result['results']['train']
				r_va = result['results']['validation']
				r_te = result['results']['test']
				df_row['train_loss'] = r_tr['loss']
				df_row['train_f1'] = r_tr['f1']
				df_row['val_loss'] = r_va['loss']
				df_row['val_f1'] = r_va['f1']
				df_row['test_loss'] = r_te['loss']
				df_row['test_f1'] = r_te['f1']
			self.data_frame = self.data_frame.append(df_row, ignore_index=True)
			logger.info('#################################################################################')
			logger.info('############################## EVALUATION COMPLETE ##############################')
			logger.info('#################################################################################')

		print('#################################################################################')
		print('############################## EXPERIMENT COMPLETE ##############################\n\n')

		f1 = 0.0
		for i, r in enumerate(results):
			print(f'Run [{i}/{self.runs}]: {r["results"]["test"]["f1"]}') 
			f1 += r['results']['test']['f1']
		print(f'------------------------------\nMean: {f1/len(results)}')
		
		# export
		try:
			self.data_frame.to_csv(e_path + 'df.csv')
		except Exception as err:
			logger.exception('Could not export dataframe to csv')

		try:
			self.data_frame.to_pickle(e_path + 'df.pkl')
		except Exception as err:
			logger.exception('Could not pickle dataframe')

		print('TEST F1 Statistics\n' + str(self.data_frame.test_f1.describe()))
		logger.info('\n' + str(self.data_frame.test_f1.describe()))
		return self.data_frame


	def _print_result(self, result, i):
		if result['status'] == STATUS_OK:
			print(f"       .---.\n \
	/     \\\n\
	\\.@-@./\t\tExperiment: [{i}/{self.runs}]\n\
	/`\\_/`\\\t\tStatus: {result['status']}\n\
	//  _  \\\\\tLoss: {result['best_loss']}\n\
	| \\     )|_\tf1: {result['best_f1']}\n\
	/`\\_`>  <_/ \\\n\
	\\__/'---'\\__/\n")
		else:
			print(f"       .---.\n \
	/     \\\n\
	\\.@-@./\tExperiment: [{i}/{self.runs}] (FAIL)\n\
	/`\\_/`\\\n\
	//  _  \\\\\\n\
	| \\     )|_\n\
	/`\\_`>  <_/ \\\n\
	\\__/'---'\\__/\n")






