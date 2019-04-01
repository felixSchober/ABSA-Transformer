import logging

import torch

from torch.autograd import *
import torchtext
from torchtext import data
from stop_words import get_stop_words

from data.torchtext.custom_fields import ReversibleField
from data.torchtext.organic_dataset import *
from data.data_loader import get_embedding

from misc.run_configuration import RunConfiguration


def preprocess_word(word: str) -> str:
	# TODO: Actual processing
	return word

def preprocess_relevance_word(word: str) -> int:
	if word == 'false':
		return 0
	return 1

def organic_dataset(
				task:str,
				pretrained_vectors,
				hyperparameters: RunConfiguration,
				batch_size=80,
				root='./bio',
				train_file='train.csv',
				validation_file='validation.csv',
				test_file='test.csv',
				use_cuda=False,
				verbose=True):

	assert task in [ORGANIC_TASK_ALL, ORGANIC_TASK_ENTITIES, ORGANIC_TASK_ATTRIBUTES, ORGANIC_TASK_ALL_COMBINE, ORGANIC_TASK_ATTRIBUTES_COMBINE, ORGANIC_TASK_ENTITIES_COMBINE, ORGANIC_TASK_COARSE, ORGANIC_TASK_COARSE_COMBINE]
	assert hyperparameters.language == 'en'

	if hyperparameters.use_stop_words:
		stop_words = get_stop_words(hyperparameters.language)
	else:
		stop_words = []

	# Sequence number
	# Index
	# Author_Id
	# Comment number
	# Sentence number
	# Domain Relevance
	# Sentiment
	# Entity
	# Attribute
	# Sentence
	# Source File
	# Aspect

	sq_num_field =  ReversibleField(
							batch_first=True,
							is_target=False,
							sequential=False,
							use_vocab=True,
							unk_token=None)

	idx_field = ReversibleField(
							batch_first=True,
							is_target=False,
							sequential=False,
							use_vocab=True,
							unk_token=None)

	sentiment_field = ReversibleField(
							batch_first=True,
							is_target=True,
							sequential=False,
							init_token=None,
							eos_token=None,
							unk_token=None,
							use_vocab=True)

	aspect_sentiment_field = ReversibleField(
							batch_first=True,
							is_target=True,
							sequential=True,
							init_token=None,
							eos_token=None,
							pad_token=None,
							unk_token=None,
							use_vocab=True)

	padding_field = data.Field(
							batch_first=True,
							sequential=True,
							fix_length=hyperparameters.clip_comments_to,
							use_vocab=True,
							init_token=None,
							eos_token=None,
							unk_token=None,
							is_target=False)

	comment_field = data.Field(
							batch_first=True,    # produce tensors with batch dimension first
							lower=True,
							fix_length=hyperparameters.clip_comments_to,
							sequential=True,
							use_vocab=True,
							init_token=None,
							eos_token=None,
							is_target=False,
							stop_words=stop_words)

	fields = [
		(None, None), 										# sq_number
		(None, None),
		(None, None),
		(None, None),
		(None, None),
		(None, None),
		('aspect_sentiments', aspect_sentiment_field),		# aspect sentiment field List of 20 aspects with positive, negative, neutral, n/a
		('comments', comment_field),                        # comment itself e.g. (@KuttnerSarah @DB_Bahn Hund = Fahrgast, Hund in Box = Gepäck.skurril, oder?)
		(None, None), 										# source file
		('padding', padding_field)                          # artificial field that we append to fill it with the padding information later to create the masks

	]

	if task in [ORGANIC_TASK_ALL, ORGANIC_TASK_ENTITIES, ORGANIC_TASK_ATTRIBUTES]:
		train, val, test = SingleSentenceOrganicDataset.splits(
									path=root,
									root='.data',
									train=train_file,
									validation=validation_file,
									test=test_file,
									separator='|',
									fields=fields,
									verbose=verbose,
									hp=hyperparameters,
									task=task)
	else:
		train, val, test = DoubleSentenceOrganicDataset.splits(
									path=root,
									root='.data',
									train=train_file,
									validation=validation_file,
									test=test_file,
									separator='|',
									fields=fields,
									verbose=verbose,
									hp=hyperparameters,
									task=task)

	# use updated fields
	fields = train.fields
	comment_field.build_vocab(train.comments, val.comments, test.comments, vectors=[pretrained_vectors])
	padding_field.build_vocab(train.padding, val.comments, test.comments)
	aspect_sentiment_field.build_vocab(train.aspect_sentiments, val.aspect_sentiments, test.aspect_sentiments)
	# id_field.build_vocab(train.id, val.id, test.id)

	# build aspect fields
	aspect_sentiment_fields = []
	for s_cat, f in train.aspect_sentiment_fields:
		f.build_vocab(train.__getattr__(s_cat), val.__getattr__(s_cat), test.__getattr__(s_cat))
		aspect_sentiment_fields.append(f)

	train_device = torch.device('cuda:0' if torch.cuda.is_available() and use_cuda else 'cpu')
	train_iter, val_iter, test_iter = data.BucketIterator.splits(
		(train, val, test), batch_size=batch_size, device=train_device)

	# add embeddings
	embedding_size = comment_field.vocab.vectors.shape[1]
	source_embedding = get_embedding(comment_field.vocab, embedding_size, hyperparameters.embedding_type)

	examples = train.examples[0:3] + val.examples[0:3] + test.examples[0:3]

	return {
		'task': 'organic19_' + task,
		'split_length': (len(train), len(val), len(test)),
		'iters': (train_iter, val_iter, test_iter), 
		'vocabs': (comment_field.vocab, aspect_sentiment_field.vocab),
		'fields': fields,
		'source_field_name': 'comments',
		'source_field': comment_field,
		'aspect_sentiment_field': aspect_sentiment_field,
		'target_field_name': 'aspect_sentiments',
		#'target': [('general_sentiments', general_sentiment_field), ('aspect_sentiments', aspect_sentiment_field)] + train.aspect_sentiment_fields,
		'target': train.aspect_sentiment_fields,
		'padding_field_name': 'padding',
		'examples': examples,
		'embeddings': (source_embedding, None),
		'dummy_input': Variable(torch.zeros((batch_size, 42), dtype=torch.long)),
		'baselines': {
			'germeval_baseline': 0.667,
			'germeval_best': 0.749
		}
		}