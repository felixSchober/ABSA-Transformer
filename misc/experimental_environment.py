import matplotlib
import copy
import logging
import time
import math
import os

from data.data_loader import Dataset

from misc.preferences import PREFERENCES
from misc.run_configuration import get_default_params, randomize_params, OutputLayerType, hyperOpt_goodParams, elmo_params, good_organic_hp_params
from misc import utils
import traceback
from optimizer import get_optimizer
from criterion import NllLoss, LossCombiner

from models.transformer.encoder import TransformerEncoder
from trainer.train import Trainer

import pprint
import torch
import pandas as pd

STATUS_FAIL = 'fail'
STATUS_OK = 'ok'

class Experiment(object):

	def __init__(self, experiment_name, experiment_description, default_hp, overwrite_hp, data_loader, runs=5):

		# make sure preferences are set
		assert PREFERENCES.data_root is not None
		assert PREFERENCES.data_train is not None
		assert PREFERENCES.data_validation is not None
		assert PREFERENCES.data_test is not None
		assert PREFERENCES.source_index is not None
		assert PREFERENCES.target_vocab_index is not None
		assert data_loader is not None
		assert runs > 0

		self.experiment_name = experiment_name
		self.experiment_description = experiment_description
		self.default_hp = default_hp
		self.overwrite_hp = overwrite_hp
		self.use_cuda = torch.cuda.is_available()
		self.dsl = data_loader
		self.runs = runs
		self.hp = None
		self.data_frame = pd.DataFrame()

		print(f'Experiment {self.experiment_name} initialized')
		

	def _initialize(self):
		# make sure the seed is not set
		self.overwrite_hp = {**self.overwrite_hp, **{'seed': None}}
		self.hp = get_default_params(use_cuda=self.use_cuda, overwrite=self.overwrite_hp, from_default=self.default_hp)
		assert self.hp.seed is None

	def load_model(self, dataset, rc, experiment_name):

		transformer = TransformerEncoder(dataset.source_embedding,
										hyperparameters=rc)

		# NER or ABSA-task?
		if rc.task == 'ner':
			from models.transformer_tagger import TransformerTagger
			from models.output_layers import SoftmaxOutputLayer
			loss = NllLoss(dataset.target_size, dataset.class_weights[0])
			softmax = SoftmaxOutputLayer(rc.model_size, dataset.target_size)
			model = TransformerTagger(transformer, softmax)

		else:
			from models.jointAspectTagger import JointAspectTagger
			loss = LossCombiner(4, dataset.class_weights, NllLoss)
			model = JointAspectTagger(transformer, rc, 4, 20, dataset.target_names)


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
		dataset = Dataset(
			task,
			logger,
			rc,
			source_index=PREFERENCES.source_index,
			target_vocab_index=PREFERENCES.target_vocab_index,
			data_path=PREFERENCES.data_root,
			train_file=PREFERENCES.data_train,
			valid_file=PREFERENCES.data_validation,
			test_file=PREFERENCES.data_test,
			file_format=PREFERENCES.file_format,
			init_token=None,
			eos_token=None
		)
		dataset.load_data(self.dsl, verbose=True)
		return dataset

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

		# time dataset loading
		ds_start = time.time()

		try:
			dataset = self.load_dataset(rc, dataset_logger, rc.task)
		except Exception as err:
			print('Could load dataset: ' + repr(err))
			print(traceback.print_tb(err.__traceback__))
			logger.exception("Could not load dataset")
			return {
				'status': STATUS_FAIL,
				'eval_time': time.time() - run_time
			}
		ds_end = time.time()

		logger.debug(f'dataset loaded. Duration: {ds_end - ds_start}')
		print(f'dataset loaded. Duration: {ds_end - ds_start}')
		logger.debug('Load model')

		try:
			trainer = self.load_model(dataset, rc, experiment_name)
		except Exception as err:
			print('Could not load model: ' + repr(err))
			print(traceback.print_tb(err.__traceback__))

			logger.exception("Could not load model")
			return {
				'status': STATUS_FAIL,
				'eval_time': time.time() - run_time
			}

		logger.debug('model loaded')

		logger.debug('Begin training')
		model = None
		try:
			tr_start = time.time()
			result = trainer.train(use_cuda=rc.use_cuda, perform_evaluation=False)
			tr_end = time.time()
			model = result['model']
		except Exception as err:
			print('Exception while training: ' + repr(err))
			print(traceback.print_tb(err.__traceback__))
			logger.exception("Could not complete iteration")
			return {
				'status': STATUS_FAIL,
				'eval_time': time.time() - run_time,
				'best_loss': trainer.get_best_loss(),
				'best_f1': trainer.get_best_f1()
			}
		logger.info(f'Training duration was {tr_end - tr_start}')
		print(f'Training duration was {tr_end - tr_start}')

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
			result = trainer.perform_final_evaluation(use_test_set=True, verbose=False, c_matrix=True)
		except Exception as err:
			logger.exception("Could not complete iteration evaluation.")
			print('Could not complete iteration evaluation: ' + repr(err))
			print(traceback.print_tb(err.__traceback__))
			return {
				'status': STATUS_FAIL,
				'eval_time': time.time() - run_time,
				'best_loss': trainer.get_best_loss(),
				'best_f1': trainer.get_best_f1()
			}
		print(f'VAL f1\t{trainer.get_best_f1()} - ({result[1][1]})')
		print(f'(macro) f1\t{trainer.get_final_macro_f1()}')

		print(f'VAL loss\t{trainer.get_best_loss()}')
		return {
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
						'f1': result[1][1],
						'f1_macro': trainer.get_final_macro_f1()['valid']
					},
					'test': {
						'loss': result[2][0],
						'f1': result[2][1],
						'f1_macro': trainer.get_final_macro_f1()['test']
					}
				}
			}

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
			result = self._objective(self.hp, i)
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
				df_row['val_f1_macro'] = r_va['f1_macro']

				df_row['test_loss'] = r_te['loss']
				df_row['test_f1'] = r_te['f1']
				df_row['test_f1_macro'] = r_te['f1_macro']

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
			logger.exception('Could not export dataframe to csv ' + repr(err))
			print(traceback.print_tb(err.__traceback__))

		try:
			self.data_frame.to_pickle(e_path + 'df.pkl')
		except Exception as err:
			logger.exception('Could not pickle dataframe ' + repr(err))
			print(traceback.print_tb(err.__traceback__))

		print('TEST MICRO F1 Statistics\n' + str(self.data_frame.test_f1.describe()))
		print('TEST MACRO F1 Statistics\n' + str(self.data_frame.test_f1_macro.describe()))

		logger.info('\n\nMICRO\n' + str(self.data_frame.test_f1.describe()))
		logger.info('\n\nMACRO\n' + str(self.data_frame.test_f1_macro.describe()))

		return (self.data_frame, e_path)


	def _print_result(self, result, i):
		if result['status'] == STATUS_OK:
			print(f".---.\n \
/     \\\n\
 \\.@-@./\t\tExperiment: [{i}/{self.runs}]\n\
 /`\\_/`\\\t\tStatus: {result['status']}\n\
 //  _  \\\\\tLoss: {result['best_loss']}\n\
 | \\     )|_\tf1: {result['best_f1']}\n\
 /`\\_`>  <_/ \\\n\
 \\__/'---'\\__/\n")
		else:
			print(f"  .---.\n \
/     \\\n\
 \\.@-@./\tExperiment: [{i}/{self.runs}] (FAIL)\n\
 /`\\_/`\\\n\
 //  _  \\\\\n\
| \\     )|_\n\
 /`\\_`>  <_/ \\\n\
 \\__/'---'\\__/\n")






