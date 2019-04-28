import logging
from misc.preferences import PREFERENCES
from misc.run_configuration import default_params, hyperOpt_goodParams
from misc import utils
from misc.experimental_environment import Experiment
from misc.transfer_learning_experiment import TransferLearningExperiment
import argparse
import traceback

def run(args, parser):
	dataset_choice = args.dataset
	runs = args.runs
	epochs = args.epochs
	name = args.name
	description = args.description
	task = args.task

	possible_dataset_values = ['germeval', 'organic', 'coNLL-2003', 'amazon']
	if dataset_choice not in possible_dataset_values:
		parser.error('The dataset argument was not in the allowed range of values: ' + str(possible_dataset_values))

	if dataset_choice == possible_dataset_values[0]:
		from data.germeval2017 import germeval2017_dataset as dsl
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
		from misc.run_configuration import hyperOpt_goodParams

		specific_hp = {**hyperOpt_goodParams, **{
			'task': task,
			'use_stop_words': True,
			'language': 'de',
			'embedding_type': 'fasttext'
		}}

	elif dataset_choice == possible_dataset_values[1]:
			from data.organic2019 import organic_dataset as dsl
			from data.organic2019 import ORGANIC_TASK_ALL, ORGANIC_TASK_ENTITIES, ORGANIC_TASK_ATTRIBUTES, ORGANIC_TASK_ENTITIES_COMBINE, ORGANIC_TASK_COARSE
			from misc.run_configuration import good_organic_hp_params

			possible_organic_values = [ORGANIC_TASK_ALL, ORGANIC_TASK_ENTITIES, ORGANIC_TASK_ATTRIBUTES, ORGANIC_TASK_ENTITIES_COMBINE, ORGANIC_TASK_COARSE]
			if task not in possible_organic_values:
				parser.error('The task argument was not in the allowed range of values: ' + str(possible_organic_values))

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
			specific_hp = good_organic_hp_params
			specific_hp['task'] = task			

	elif dataset_choice == possible_dataset_values[2]:
		PREFERENCES.defaults(

			data_root='./data/data/conll2003',
			data_train='eng.train.txt',
			data_validation='eng.testa.txt',
			data_test='eng.testb.txt',
			source_index=0,
			target_vocab_index=1,
			file_format='txt',
			language='en'
		)
		from data.conll import conll2003_dataset as dsl
		from misc.run_configuration import hyperOpt_goodParams

		specific_hp = {**hyperOpt_goodParams, **{
			'task': 'ner',
			'language': 'en',
			'embedding_type': 'fasttext'
		}}
	
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
		from misc.run_configuration import hyperOpt_goodParams

		specific_hp = {**hyperOpt_goodParams, **{
			'task': 'amazon',
			'use_spell_checkers': True,
			'use_stop_words': True,
			'language': 'en',
			'clip_comments_to': 100,
			'embedding_type': 'fasttext'
		}}

	main_experiment_name = name
	use_cuda = True
	experiment_name = utils.create_loggers(experiment_name=main_experiment_name)
	logger = logging.getLogger(__name__)
	dataset_logger = logging.getLogger('data_loader')
	logger.info('Run hyper parameter random grid search for experiment with name ' + main_experiment_name)
	logger.info('num_optim_iterations: ' + str(runs))
	specific_hp['epochs'] = epochs

	try:
		logger.info('Current commit: ' + utils.get_current_git_commit())
		print('Current commit: ' + utils.get_current_git_commit())
	except Exception as err:
		logger.exception('Could not print current commit')

	try:
		e = Experiment(name, description, default_params, specific_hp, dsl, runs=runs)
		e.run()
	except Exception as err:
		logger.exception('Could not complete run')
		print('Could not complete run. The log file provides more details.')
		print(repr(err))
		traceback.print_tb(err.__traceback__)

if __name__ == "__main__":
	parser = argparse.ArgumentParser(description='HyperOpt hp optimization tool')
	parser.add_argument('dataset', type=str,
						help='Specify which dataset to optimize')
	parser.add_argument('--runs', type=int, default=1,
						help='Number of runs evaluation runs to perform')
	parser.add_argument('--epochs', type=int, default=35,
						help='Number of epochs to perform')
	parser.add_argument('--name', type=str, default='test',
						help='Specify a name of the optimization run')
	parser.add_argument('--description', type=str, default='test run on {} with {} epochs and {} validations',
						help='Specify a name of the optimization run')
	parser.add_argument('--task', type=str,
						help='Specify the task to execute. Only applicable when using the organic dataset')
	args = parser.parse_args()

	run(args, parser)
	print('Exit')