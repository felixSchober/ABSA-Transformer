import logging
import torch
import os

from data.data_loader import Dataset
from data.organic2019 import organic_dataset as dsl
from misc.preferences import PREFERENCES
from misc.visualizer import *
from misc.run_configuration import get_default_params, randomize_params, OutputLayerType, hyperOpt_goodParams, elmo_params, good_organic_hp_params, default_params
from misc import utils

from optimizer import get_optimizer
from criterion import NllLoss, LossCombiner

from models.transformer.encoder import TransformerEncoder
from models.jointAspectTagger import JointAspectTagger
from trainer.train import Trainer, create_padding_masks
import pprint
import pickle
import torchtext
import pandas as pd


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


def write_evaluation_file(iterator: torchtext.data.Iterator, dataset: Dataset, trainer: Trainer, filename='prediction.xml'):
	fields = dataset.fields
	all_predictions = []
	all_targets = []
	with torch.no_grad():
		iterator.init_epoch()
		
		df = {
			'Author_ID': [],
			'Comment_number': [],
			'Sentence_number': [],
			'Sentiment': [],
			'Entity': [],
			'Attribute': [],
			'Aspect': [],
			'Sentence': [],
			'Domain_Relevance': [],
			'id': []
		}

		df_gold = prepare_gold_labels()

		# metrics for aspect + sentiment
		tp = 0
		fp = 0
		fn = 0

		# metrics for aspect
		tp_a = 0
		fp_a = 0
		fn_a = 0

		for batch in iterator:
			comment_id, comment, target_aspect_sentiment, padding = batch.id, batch.comments, batch.aspect_sentiments, batch.padding
			comment_id = fields['id'].reverse(comment_id.unsqueeze(1))
			comment_decoded = [get_gold_label_row(df_gold, c_id)['Sentence'] for c_id in comment_id]

			source_mask = create_padding_masks(padding, 1)
			prediction = trainer.model.predict(comment, source_mask)

			all_predictions.append(prediction)
			all_targets.append(target_aspect_sentiment)

			p = torch.t(prediction)
			t = torch.t(target_aspect_sentiment)

			for a_i in range(dataset.target_size):

				# for aspect match it only has to predict "some" sentiment
				p_mask = p[a_i] > 0
				t_mask = t[a_i] > 0
				c_matrix = confusion_matrix(t_mask.cpu(), p_mask.cpu(), labels=[1, 0])
				tp_a += c_matrix[0,0]
				fp_a += c_matrix[0,1]
				fn_a += c_matrix[1,0]

				for s_i in range(4):

					if s_i == 0:
						continue
					p_mask = p[a_i] == s_i
					t_mask = t[a_i] == s_i
					c_matrix = confusion_matrix(t_mask.cpu(), p_mask.cpu(), labels=[1, 0])
					tp += c_matrix[0,0]
					fp += c_matrix[0,1]
					fn += c_matrix[1,0]

			aspect_sentiment = fields['aspect_sentiments'].reverse(prediction, detokenize=False)

			for i in range(len(comment_id)):

				c_id = comment_id[i].split('_')
				a_id = c_id[0]
				c_num = c_id[1]
				s_num = c_id[2]


				# add aspects
				for sentiment, a_name in zip(aspect_sentiment[i], dataset.target_names):
					if sentiment == 'n/a':
						continue

					(entity, attribute) = a_name.split(': ')


					df['Author_ID'].append(a_i)
					df['Comment_number'].append(c_num)
					df['Sentence_number'].append(s_num)
					df['Sentence'].append(comment_decoded[i])
					df['Aspect'].append(f'{entity}-{attribute}')
					df['Sentiment'].append(sentiment)
					df['Entity'].append(entity)
					df['Attribute'].append(attribute)
					df['Domain_Relevance'].append('9')
					df['id'].append(comment_id[i])

		# add not relevance labels
		not_relevant = get_not_relevant_labels(df_gold)[['id', 'Author_ID', 'Comment_number', 'Sentence_number', 'Sentence','Aspect', 'Sentiment', 'Entity', 'Attribute', 'Domain_Relevance']]
		df = pd.DataFrame(df)
		df = df.append(not_relevant, ignore_index=True)
		df = df.sort_values(by=['id'])

		df.to_csv(os.path.join(os.getcwd(), 'evaluation', filename), sep='|')

		print(f'TP - Sentiment + Aspect: {tp}')
		print(f'FP - Sentiment + Aspect: {fp}')
		print(f'FN - Sentiment + Aspect: {fn}')

		precision = float(tp) / (tp + fp)
		recall = float(tp) / (tp + fn)
		f1 = 2.0 * precision * recall / (precision + recall)
		print(f'F1 - Sentiment + Aspect: {f1}')

		print(f'TP - Aspect: {tp_a}')
		print(f'FP - Aspect: {fp_a}')
		print(f'FN - Aspect: {fn_a}')

		precision = float(tp_a) / (tp_a + fp_a)
		recall = float(tp_a) / (tp_a + fn_a)
		f1 = 2.0 * precision * recall / (precision + recall)
		print(f'F1 - Aspect: {f1}')

		# with open('all_predictions.pkl', 'wb') as f:
		# 	pickle.dump(all_predictions, f) 

		# with open('all_targets.pkl', 'wb') as f:
		# 	pickle.dump(all_targets, f) 


def prepare_gold_labels():
	path = os.path.join(os.getcwd(), 'data', 'data', 'organic2019', 'test.csv')
	df = pd.read_csv(path, sep='|')
	df['id'] = df.apply(lambda r: f'{r["Author_ID"]}_{r["Comment_number"]}_{r["Sentence_number"]}', axis=1)
	return df

def get_gold_label_row(df_gold, comment_id):
	return df_gold[df_gold['id']==comment_id]

def get_not_relevant_labels(df_gold):
	return df_gold[df_gold['Domain_Relevance'] == 0]

# experiment_name = utils.create_loggers(experiment_name='testing')
# logger = logging.getLogger(__name__)

# default_hp = get_default_params(False)

# logger.info(default_hp)
# print(default_hp)

# dataset = load(default_hp, logger)
# produce_test_gold_labels(dataset.test_iter, dataset)

experiment_name = 'EvaluationTest'
use_cuda = True
experiment_name = utils.create_loggers(experiment_name=experiment_name)
logger = logging.getLogger(__name__)

baseline = {**default_params, **good_organic_hp_params}
test_params = {**baseline, **{'task': 'all', 'log_every_xth_iteration': -1}}

rc = get_default_params(use_cuda=True, overwrite={}, from_default=test_params)
logger = logging.getLogger(__name__)

dataset_logger = logging.getLogger('data_loader')
logger.debug('Load dataset')

path = os.path.join(os.getcwd(), 'evaluation')
utils.create_dir_if_necessary(path)

f1_scores_test = []
f1_scores_val = []

for i in range(8):
	print('New Iteration')
	dataset = load_dataset(rc, dataset_logger, rc.task)

	logger.debug('dataset loaded')
	logger.debug('Load model')
	trainer = load_model(dataset, rc, experiment_name)
	logger.debug('model loaded')

	trainer.train(perform_evaluation=False)
	result = trainer.perform_final_evaluation(use_test_set=True, verbose=False)
	f1_scores_val.append(result[1][1])
	f1_scores_test.append(result[2][1])
	print('Write Evaluation file')
	write_evaluation_file(dataset.test_iter, dataset, trainer, filename=f'predictions_{i}.csv')

for i in range(len(f1_scores_test)):
	print(f'{i}:\tVal: {f1_scores_val[i]} - Test: {f1_scores_test[i]}')

print('Finished')