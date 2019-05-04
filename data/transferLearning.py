import logging

import torch

from torch.autograd import *
import torchtext
from torchtext import data
from stop_words import get_stop_words

from data.torchtext.custom_fields import ReversibleField
from data.torchtext.amazon_dataset import *
from data.data_loader import get_embedding

from misc.run_configuration import RunConfiguration

def transfer_learning(
				task:str,
				pretrained_vectors,
				hyperparameters: RunConfiguration,
				batch_size=80,
				loaders=[],
				roots=['./amazon'],
				train_files=['train.pkl'],
				validation_files=['validation.pkl'],
				test_files=['test.pkl'],
				use_cuda=True,
				verbose=True):

	logger = logging.getLogger(__name__)

	assert len(loaders) == len(roots) == len(train_files) == len(validation_files) == len(test_files)


	loader_results = []
	fields = []
	for i, l in enumerate(loaders):
		root = roots[i]
		train_file = train_files[i]
		validation_file = validation_files[i]
		test_file = test_files[i]
		logger.info(f'[{i}/{len(loaders)}] Loading dataset with root {root}')
		result = l(
			task,
			hyperparameters,
			root,
			train_file,
			validation_file,
			test_file,
			verbose)
		logger.debug(f'Dataset {i} is loaded')
		loader_results.append(result)
		fields.extend(result['splits'][0].fields)

	all_comments = [field.comments
	for splits in loader_results
	for field in splits['splits']]

	all_paddings = [field.padding
	for splits in loader_results
	for field in splits['splits']]

	# build vocab and iterators
	train_device = torch.device('cuda:0' if torch.cuda.is_available() and use_cuda else 'cpu')

	datasets = {
				'stats': [],
				'split_length': [],
				'vocabs': [],
				'source_field': [],
				'aspect_sentiment_field': [],
				'target': [],
				'examples': [],
				'iters': []
			}
	comment_vocab = None
	for r in loader_results:
		train, val, test = r['splits']
		if comment_vocab is None:
			r['fields']['comment'].build_vocab(
				*all_comments,
				vectors=[pretrained_vectors],
				min_freq=3,
				max_size=40000)
			comment_vocab = r['fields']['comment'].vocab
		else:
			r['fields']['comment'].vocab = comment_vocab
		r['fields']['padding'].build_vocab(*all_paddings)
		r['fields']['aspect_sentiment'].build_vocab(train.aspect_sentiments, val.aspect_sentiments, test.aspect_sentiments)

		examples = train.examples[0:3] + val.examples[0:3] + test.examples[0:3]

		iters = data.BucketIterator.splits(
		(train, val, test), batch_size=batch_size, device=train_device, shuffle=True)

		datasets['stats'].append((train.stats, val.stats, test.stats))
		datasets['split_length'].append((len(train), len(val), len(test)))
		datasets['vocabs'].append((r['fields']['comment'].vocab, r['fields']['aspect_sentiment'].vocab))
		datasets['source_field'].append(r['fields']['comment'])
		datasets['aspect_sentiment_field'].append(r['fields']['aspect_sentiment'])
		datasets['target'].append(train.aspect_sentiment_fields)
		datasets['examples'].append(examples)
		datasets['iters'].append(iters)


		# build aspect fields
		aspect_sentiment_fields = []
		for s_cat, f in train.aspect_sentiment_fields:
			f.build_vocab(train.__getattr__(s_cat), val.__getattr__(s_cat), test.__getattr__(s_cat))
			aspect_sentiment_fields.append(f)

	

	# add embeddings
	embedding_size = datasets['source_field'][0].vocab.vectors.shape[1]
	source_embedding = get_embedding(datasets['source_field'][0].vocab, embedding_size, hyperparameters.embedding_type)

	result = {
		'task': 'amazon',
		'fields': fields,
		'source_field_name': 'comments',
		'target_field_name': 'aspect_sentiments',
		'padding_field_name': 'padding',
		'embeddings': (source_embedding, None),
		'dummy_input': Variable(torch.zeros((batch_size, 42), dtype=torch.long)),
		'baselines': {
			'germeval_baseline': 0.667,
			'germeval_best': 0.749
		}
		}

	return {**datasets, **result}