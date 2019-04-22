
import numpy as np
import math
import os
import time
import logging
from hyperopt.plotting import *
from hyperopt import fmin, tpe, hp, STATUS_OK, STATUS_FAIL, Trials, base
from data.data_loader import Dataset
from misc.preferences import PREFERENCES
from misc.run_configuration import from_hyperopt, OutputLayerType, LearningSchedulerType, OptimizerType, default_params
from misc import utils
from misc.hyperopt_space import *

from optimizer import get_optimizer
from criterion import NllLoss, LossCombiner
from models.transformer.encoder import TransformerEncoder
from models.jointAspectTagger import JointAspectTagger
from trainer.train import Trainer
import pprint
import argparse
import pickle

def load_model(dataset, rc, experiment_name):
	loss = LossCombiner(4, dataset.class_weights, NllLoss)
	transformer = TransformerEncoder(dataset.source_embedding,
									 hyperparameters=rc)
	model = JointAspectTagger(transformer, rc, 4, 20, dataset.target_names)
	optimizer = get_optimizer(model, rc)
	trainer = Trainer(
						model,
						loss,
						optimizer,
						rc,
						dataset,
						experiment_name,
						enable_tensorboard=False,
						verbose=False)
	return trainer

def load_dataset(rc, logger, task):
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
	dataset.load_data(dsl, verbose=False)
	return dataset

def objective(parameters):
	run_time = time.time()
	
	utils.reset_loggers()
	experiment_name = utils.create_loggers(experiment_name=main_experiment_name)
	logger = logging.getLogger(__name__)
	dataset_logger = logging.getLogger('data_loader')

	# generate hp's from parameters
	try:
		rc = from_hyperopt(parameters, use_cuda, model_size=300, early_stopping=5, num_epochs=35, log_every_xth_iteration=-1, language=PREFERENCES.language)
	except Exception as err:
		print('Could not convert params: ' + str(err))
		logger.exception("Could not load parameters from hyperopt configuration: " + parameters)
		return {
			'status': STATUS_FAIL,
			'eval_time': time.time() - run_time
		}
	logger.info('New Params:')
	logger.info(rc)
	print('\n\n#########################################################################')
	print(rc)

	logger.debug('Load dataset')
	try:
		dataset = load_dataset(rc, dataset_logger, rc.task)
	except Exception as err:
		print('Could not load dataset: ' + str(err))
		logger.exception("Could not load dataset")
		return {
			'status': STATUS_FAIL,
			'eval_time': time.time() - run_time
		}
	logger.debug('dataset loaded')
	logger.debug('Load model')

	try:
		trainer = load_model(dataset, rc, experiment_name)
	except Exception as err:
		print('Could not load model: ' + str(err))
		logger.exception("Could not load model")
		return {
			'status': STATUS_FAIL,
			'eval_time': time.time() - run_time
		}

	logger.debug('model loaded')

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
	
	print(f"       .---.\n \
		 /     \\\n\
		  \\.@-@./\n\
		  /`\\_/`\\\n\
		 //  _  \\\\\tLoss: {trainer.get_best_loss()}\n\
		| \\     )|_\tf1: {trainer.get_best_f1()}\n\
	   /`\\_`>  <_/ \\\n\
	   \\__/'---'\\__/\n")
	
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
					'f1': result[1][1]
				},
				'test': {
					'loss': result[2][0],
					'f1': result[2][1]
				}
			}
		}

parser = argparse.ArgumentParser(description='HyperOpt hp optimization tool')

parser.add_argument('dataset', type=str,
					help='Specify which dataset to optimize')

parser.add_argument('--runs', type=int,
					help='Number of runs to perform')

parser.add_argument('--name', type=str, default='HyperOpt',
					help='Specify a name of the optimization run')

parser.add_argument('--description', type=str, default='HyperOpt run',
					help='Specify a description of the optimization run')

args = parser.parse_args()

possible_dataset_values = ['germeval', 'organic', 'amazon']
dataset_choice = args.dataset
if dataset_choice not in possible_dataset_values:
	parser.error('The dataset argument was not in the allowed range of values: ' + str(possible_dataset_values))

runs = args.runs

if dataset_choice == possible_dataset_values[0]:
	PREFERENCES.defaults(
		data_root='./data/data/germeval2017',
		data_train='train_v1.4.tsv',    
		data_validation='dev_v1.4.tsv',
		data_test='test_TIMESTAMP1.tsv',
		source_index=0,
		target_vocab_index=2,
		file_format='csv',
		language='de'
	)
	from data.germeval2017 import germeval2017_dataset as dsl

	search_space = {
		'batch_size': hp.quniform('batch_size', 10, 100, 1),
		'num_encoder_blocks': hp.quniform('num_encoder_blocks', 1, 8, 1),
		'pointwise_layer_size': hp.quniform('pointwise_layer_size', 32, 256, 1),
		'clip_comments_to': hp.quniform('clip_comments_to', 10, 250, 1),
		'dropout_rate': hp.uniform('dropout_rate', 0.0, 0.8),
		'output_dropout_rate': hp.uniform('last_layer_dropout', 0.0, 0.8),
		'num_heads': hp.choice('num_heads', [1, 2, 3, 4, 5]),
		'transformer_use_bias': hp_bool('transformer_use_bias'),
		'output_layer': hp.choice('output_layer', [
			{
				'type': OutputLayerType.Convolutions,
				'output_conv_num_filters': hp.quniform('output_conv_num_filters', 1, 400, 1),
				'output_conv_kernel_size': hp.quniform('output_conv_kernel_size', 1, 10, 1),
				'output_conv_stride': hp.quniform('output_conv_stride', 1, 10, 1),
				'output_conv_padding': hp.quniform('output_conv_padding', 0, 5, 1),
			},
			{
				'type': OutputLayerType.LinearSum
			}
		]),
		'learning_rate_scheduler': hp.choice('learning_rate_scheduler', [
			{
				'type': LearningSchedulerType.Noam,
				'noam_learning_rate_warmup': hp.quniform('noam_learning_rate_warmup', 1000, 9000, 1),
				'noam_learning_rate_factor': hp.uniform('noam_learning_rate_factor', 0.01, 4)
			}
		]),
		'optimizer': hp.choice('optimizer', [
			{
				'type': OptimizerType.Adam,
				'adam_beta1': hp.uniform('adam_beta1', 0.7, 0.999),
				'adam_beta2': hp.uniform('adam_beta2', 0.7, 0.999),
				'adam_eps': hp.loguniform('adam_eps', np.log(1e-10), np.log(1)),
				'learning_rate': hp.lognormal('adam_learning_rate', np.log(0.01), np.log(10)),
				'adam_weight_decay': 1*10**hp.quniform('adam_weight_decay', -8, -3, 1)
			},
			#{
			#    'type': OptimizerType.SGD,
			#    'sgd_momentum': hp.uniform('sgd_momentum', 0.4, 1),
			#    'sgd_weight_decay': hp.loguniform('sgd_weight_decay', np.log(1e-4), np.log(1)),
			#    'sgd_nesterov': hp_bool('sgd_nesterov'),
			#    'learning_rate': hp.lognormal('sgd_learning_rate', np.log(0.01), np.log(10))
		]),
		'replace_url_tokens': hp_bool('replace_url_tokens'),
		'harmonize_bahn': hp_bool('harmonize_bahn'),
		'embedding_type': hp.choice('embedding_type', ['fasttext', 'glove']),
		'embedding_name': hp.choice('embedding_name', ['6B']),
		'embedding_dim': hp.choice('embedding_dim', [300]),
		'use_stop_words': hp_bool('use_stop_words'),
		'use_spell_checker': hp_bool('use_spell_checker'),
		'embedding_type': hp.choice('embedding_type', ['fasttext', 'glove']),
		'task': 'germeval'
	}

elif dataset_choice == possible_dataset_values[1]:
	 from data.organic2019 import organic_dataset as dsl
	 from data.organic2019 import ORGANIC_TASK_ALL, ORGANIC_TASK_ENTITIES, ORGANIC_TASK_ATTRIBUTES, ORGANIC_TASK_ENTITIES_COMBINE, ORGANIC_TASK_COARSE
	 PREFERENCES.defaults(
		data_root='./data/data/organic2019',
		data_train='train.csv',    
		data_validation='validation.csv',
		data_test='test.csv',
		source_index=0,
		target_vocab_index=1,
		file_format='csv',
		language='en'
	 )

	 search_space = {
		'batch_size': hp.quniform('batch_size', 10, 64, 1),
		'num_encoder_blocks': hp.quniform('num_encoder_blocks', 1, 4, 1),
		'pointwise_layer_size': hp.quniform('pointwise_layer_size', 32, 350, 1),
		'clip_comments_to': hp.quniform('clip_comments_to', 45, 180, 1),
		'dropout_rate': hp.uniform('dropout_rate', 0.0, 0.8),
		'output_dropout_rate': hp.uniform('last_layer_dropout', 0.0, 0.8),
		'num_heads': hp.choice('num_heads', [1, 2, 3, 4, 5]),
		'transformer_use_bias': hp_bool('transformer_use_bias'),
		'output_layer': hp.choice('output_layer', [
			{
				'type': OutputLayerType.Convolutions,
				'output_conv_num_filters': hp.quniform('output_conv_num_filters', 10, 400, 1),
				'output_conv_kernel_size': hp.quniform('output_conv_kernel_size', 1, 10, 1),
				'output_conv_stride': hp.quniform('output_conv_stride', 1, 10, 1),
				'output_conv_padding': hp.quniform('output_conv_padding', 0, 5, 1),
			},
			{
				'type': OutputLayerType.LinearSum
			}
		]),
		'learning_rate_scheduler': hp.choice('learning_rate_scheduler', [
			{
				'type': LearningSchedulerType.Noam,
				'noam_learning_rate_warmup': hp.quniform('noam_learning_rate_warmup', 1000, 9000, 1),
				'noam_learning_rate_factor': hp.uniform('noam_learning_rate_factor', 0.01, 4)
			}
		]),
		'optimizer': hp.choice('optimizer', [
			{
				'type': OptimizerType.Adam,
				'adam_beta1': hp.uniform('adam_beta1', 0.7, 0.999),
				'adam_beta2': hp.uniform('adam_beta2', 0.7, 0.999),
				'adam_eps': hp.loguniform('adam_eps', np.log(1e-10), np.log(1)),
				'learning_rate': hp.lognormal('adam_learning_rate', np.log(0.01), np.log(10)),
				'adam_weight_decay': 1*10**hp.quniform('adam_weight_decay', -8, -3, 1)
			},
			#{
			#    'type': OptimizerType.SGD,
			#    'sgd_momentum': hp.uniform('sgd_momentum', 0.4, 1),
			#    'sgd_weight_decay': hp.loguniform('sgd_weight_decay', np.log(1e-4), np.log(1)),
			#    'sgd_nesterov': hp_bool('sgd_nesterov'),
			#    'learning_rate': hp.lognormal('sgd_learning_rate', np.log(0.01), np.log(10))
		]),
		'task': hp.choice('task', [
			ORGANIC_TASK_ENTITIES,
			ORGANIC_TASK_ENTITIES_COMBINE
		]),
		'use_stop_words': hp_bool('use_stop_words'),
		'use_spell_checker': hp_bool('use_spell_checker'),
		'embedding_type': hp.choice('embedding_type', ['fasttext', 'glove'])
	}
else:
	PREFERENCES.defaults(
		data_root='./data/data/amazon/splits',
		data_train='train.pkl',    
		data_validation='val.pkl',
		data_test='test.pkl',
		source_index=0,
		target_vocab_index=1,
		file_format='pkl',
		language='en'
	)
	from data.amazon import amazon_dataset as dsl


main_experiment_name = args.name
use_cuda = True
experiment_name = utils.create_loggers(experiment_name=main_experiment_name)
logger = logging.getLogger(__name__)
dataset_logger = logging.getLogger('data_loader')
logger.info('Run hyper parameter random grid search for experiment with name ' + main_experiment_name)
logger.info('num_optim_iterations: ' + str(runs))

try:
	logger.info('Current commit: ' + utils.get_current_git_commit())
	print('Current commit: ' + utils.get_current_git_commit())
except Exception as err:
	logger.exception('Could not print current commit')

trials = Trials()
try:

	best = fmin(objective,
		space=search_space,
		algo=tpe.suggest,
		max_evals=runs,
		trials=trials)

	print(best)
except Exception as err:
	logger.exception('Could not complete optimization')
	print('Could not complete optimization. The log file provides more details.')


path = os.path.join(os.getcwd(), 'logs', f'hp_run_{main_experiment_name}.pkl')
with open(path, 'wb') as f:
	pickle.dump(trials, f)
