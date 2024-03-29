import logging
import torch

from data.data_loader import Dataset
from data.germeval2017 import germeval2017_dataset as dsl
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
import xml.etree.ElementTree as ET
from bs4 import BeautifulSoup

PREFERENCES.defaults(
	data_root='./data/data/germeval2017',
	data_train='train_v1.4.tsv',    
	data_validation='dev_v1.4.tsv',
	data_test='test_TIMESTAMP1.tsv',
	source_index=0,
	target_vocab_index=2,
	file_format='csv'
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

def produce_test_gold_labels(iterator: torchtext.data.Iterator, dataset: Dataset, filename='gold_labels.xml'):

	fields = dataset.fields
	with torch.no_grad():
		iterator.init_epoch()
		
		tree = ET.ElementTree()
		root = ET.Element('Documents')

		for batch in iterator:
			doc_id, comment, relevance, aspect_sentiment, general_sentiment = batch.id, batch.comments, batch.relevance, batch.aspect_sentiments, batch.general_sentiments
			doc_id = fields['id'].reverse(doc_id.unsqueeze(1))
			comment = fields['comments'].reverse(comment)
			relevance = ['false' if r == 0 else 'true' for r in relevance]
			general_sentiment = fields['general_sentiments'].reverse(general_sentiment.unsqueeze(1))
			aspect_sentiment = fields['aspect_sentiments'].reverse(aspect_sentiment, detokenize=False)

			for i in range(len(doc_id)):
				docuement_elem = ET.SubElement(root, 'Document', {'id': doc_id[i]})

				rel_field = ET.SubElement(docuement_elem, 'relevance')
				rel_field.text = relevance[i]

				sen_field = ET.SubElement(docuement_elem, 'sentiment')
				sen_field.text = general_sentiment[i]

				text_field = ET.SubElement(docuement_elem, 'text')
				text_field.text = comment[i]

				# don't add aspects if field not relevant
				# if relevance[i] == 'false':
				# 	continue
				options_elem = ET.SubElement(docuement_elem, 'Opinions')

				# add aspects
				for sentiment, a_name in zip(aspect_sentiment[i], dataset.target_names):
					if sentiment == 'n/a':
						continue

					asp_field = ET.SubElement(options_elem, 'Opinion', {
						'category': a_name,
						'polarity': sentiment
					})

		#print(BeautifulSoup(ET.tostring(tree), "xml").prettify())
		tree._setroot(root)
		tree.write(filename, encoding='utf-8')

def write_evaluation_file(iterator: torchtext.data.Iterator, dataset: Dataset, trainer: Trainer, filename='prediction.xml'):
	fields = dataset.fields
	all_predictions = []
	all_targets = []
	with torch.no_grad():
		iterator.init_epoch()
		
		tree = ET.ElementTree()
		root = ET.Element('Documents')

		# metrics for aspect + sentiment
		tp = 0
		fp = 0
		fn = 0

		# metrics for aspect
		tp_a = 0
		fp_a = 0
		fn_a = 0

		for batch in iterator:
			doc_id, comment, relevance, target_aspect_sentiment, general_sentiment, padding = batch.id, batch.comments, batch.relevance, batch.aspect_sentiments, batch.general_sentiments, batch.padding
			doc_id = fields['id'].reverse(doc_id.unsqueeze(1))
			comment_decoded = fields['comments'].reverse(comment)
			relevance = ['false' if r == 0 else 'true' for r in relevance]
			general_sentiment = fields['general_sentiments'].reverse(general_sentiment.unsqueeze(1))

			source_mask = create_padding_masks(padding, 1)
			prediction = trainer.model.predict(comment, source_mask)

			all_predictions.append(prediction)
			all_targets.append(target_aspect_sentiment)

			p = torch.t(prediction)
			t = torch.t(target_aspect_sentiment)

			for a_i in range(20):

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

			for i in range(len(doc_id)):
				docuement_elem = ET.SubElement(root, 'Document', {'id': doc_id[i]})

				rel_field = ET.SubElement(docuement_elem, 'relevance')
				rel_field.text = relevance[i]

				sen_field = ET.SubElement(docuement_elem, 'sentiment')
				sen_field.text = general_sentiment[i]

				text_field = ET.SubElement(docuement_elem, 'text')
				text_field.text = comment_decoded[i]

				# don't add aspects if field not relevant
				# if relevance[i] == 'false':
				# 	continue
				options_elem = ET.SubElement(docuement_elem, 'Opinions')

				# add aspects
				for sentiment, a_name in zip(aspect_sentiment[i], dataset.target_names):
					if sentiment == 'n/a':
						continue

					asp_field = ET.SubElement(options_elem, 'Opinion', {
						'category': a_name,
						'polarity': sentiment
					})

		#print(BeautifulSoup(ET.tostring(tree), "xml").prettify())
		tree._setroot(root)
		tree.write(filename, encoding='utf-8')

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

		with open('all_predictions.pkl', 'wb') as f:
			pickle.dump(all_predictions, f) 

		with open('all_targets.pkl', 'wb') as f:
			pickle.dump(all_targets, f) 

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

baseline = {**default_params, **hyperOpt_goodParams}
test_params = {**baseline, **{'num_epochs': 1, 'language': 'de', 'batch_size': 45, 'task': 'germeval', 'token_removal_2': True, 'log_every_xth_iteration': -1}}

rc = get_default_params(use_cuda=True, overwrite={}, from_default=test_params)
logger = logging.getLogger(__name__)

dataset_logger = logging.getLogger('data_loader')
logger.debug('Load dataset')
dataset = load_dataset(rc, dataset_logger, rc.task)

logger.debug('dataset loaded')
logger.debug('Load model')
trainer = load_model(dataset, rc, experiment_name)
logger.debug('model loaded')


trainer.load_model(custom_path='/Users/felix/Documents/Repositories/TUM/ABSA-Transformer/logs/t3st/20190421/13/checkpoints')
trainer.set_cuda(True)
#result = trainer.perform_final_evaluation(use_test_set=True, verbose=False)

# import os
# path = os.path.join(os.getcwd(), 'logs', 'GoodResults')
# print(path)
# trainer.load_model(custom_path=path)
# trainer.set_cuda(True)
write_evaluation_file(dataset.test_iter, dataset, trainer)
produce_test_gold_labels(dataset.test_iter, dataset)
print('Finished')