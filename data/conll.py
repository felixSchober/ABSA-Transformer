from typing import Tuple, List, Dict, Optional, Union, Iterable
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import *

import torchtext
from torchtext import data
from torchtext.datasets import SequenceTaggingDataset, CoNLL2000Chunking
from torchtext.vocab import Vectors, GloVe, CharNGram
import numpy as np
import random
import logging
import spacy
from data.data_loader import get_embedding
from data.torchtext.custom_fields import ReversibleField
from data.torchtext.sequence_tagging_dataset import CustomSequenceTaggingDataSet

logger = logging.getLogger(__name__)

ExamplePair = Tuple[str, str] # x, y
ExampleSentence = List[ExamplePair]
ExampleList = List[ExampleSentence]

NER_TASK = 'ner'

def preprocess_test(word):
	return '0' if word.isdigit() else word

def conll2003_dataset(task:str,
					pretrained_vectors,
					hyperparameters,
					batch_size=80,
					root='./germeval2017',
					train_file='train_v1.4.tsv',
					validation_file='dev_v1.4.tsv',
					test_file=None,
					use_cuda=False,
					verbose=True):
	"""
	https://github.com/kolloldas/torchnlp/blob/master/torchnlp/data/conll.py
	conll2003: Conll 2003 (Parser only. You must place the files)
	Extract Conll2003 dataset using torchtext. Applies GloVe 6B.200d and Char N-gram
	pretrained vectors. Also sets up per word character Field
	Parameters:
		task: Type of tag to pick as task [pos, chunk, ner]
		batch_size: Batch size to return from iterator
		root: Dataset root directory
		train_file: Train filename
		validation_file: Validation filename
		test_file: Test filename
		convert_digits: If True will convert numbers to single 0's
	Returns:
		A dict containing:
			task: 'conll2003.' + task
			iters: (train iter, validation iter, test iter)
			vocabs: (Inputs word vocabulary, Inputs character vocabulary, 
					Tag vocabulary )
	"""
	
	# Setup fields with batch dimension first
	comments_field = ReversibleField(batch_first=True, lower=True,
								preprocessing=data.Pipeline(preprocess_test))           

	# the label constits of three parts:
	#   - Part of speech tag
	#   - syntactic chunk tag   (I-TYPE)
	#   - named entity tag      (I-TYPE)
	aspect_sentiments = ReversibleField(batch_first=True, is_target=True)

	words_field = [('comments', comments_field)]
	labels_field = [('aspect_sentiments', aspect_sentiments) if label == task else (None, None) 
				for label in ['pos', 'chunk', 'ner']]
	fields = ( words_field + labels_field)

	# Load the data
	train, val, test = CustomSequenceTaggingDataSet.splits(
								path=root, 
								train=train_file, 
								validation=validation_file, 
								test=test_file,
								separator=' ',
								fields=tuple(fields))

	logger.info('---------- CONLL 2003 %s ---------'%task)
	logger.info('Train size: %d'%(len(train)))
	logger.info('Validation size: %d'%(len(val)))
	logger.info('Test size: %d'%(len(test)))
	
	# Build vocab
	# inputs_char.build_vocab(train.inputs_char, val.inputs_char, test.inputs_char)
	
	comments_field.build_vocab(train.comments, val.comments, test.comments, max_size=50000,
						vectors=[pretrained_vectors])
	
	aspect_sentiments.build_vocab(train.aspect_sentiments)
	logger.info('Input vocab size:%d'%(len(comments_field.vocab)))
	logger.info('Tagset size: %d'%(len(aspect_sentiments.vocab)))

	# Get iterators
	train_iter, val_iter, test_iter = data.BucketIterator.splits(
							(train, val, test), batch_size=batch_size, shuffle=True,
							device=torch.device("cuda:0" if torch.cuda.is_available() and use_cuda else "cpu"))
	train_iter.repeat = False
	val_iter.repeat = False

	# add embeddings
	embedding_size = comments_field.vocab.vectors.shape[1]
	source_embedding = get_embedding(comments_field.vocab, embedding_size, 'glove')
	
	examples = train.examples[0:3] + val.examples[0:3] + test.examples[0:3]
	# return {
	#     'task': 'conll2003.%s'%task,
	#     'iters': (train_iter, val_iter, test_iter), 
	#     'vocabs': (comments.vocab, aspect_sentiments.vocab),
	#     'word_field': comments,
	#     'examples': examples,
	#     'embeddings': (source_embedding, None),
	#     'dummy_input': Variable(torch.zeros((batch_size, 42), dtype=torch.long))
	#     }
	return {
		'task': task,
		'stats': None,
		'split_length': (len(train), len(val), len(test)),
		'iters': (train_iter, val_iter, test_iter), 
		'vocabs': (comments_field.vocab, aspect_sentiments.vocab),
		'fields': {'comments': comments_field, 'aspect_sentiments': aspect_sentiments},
		'source_field_name': 'comments',
		'source_field': comments_field,
		'aspect_sentiment_field': aspect_sentiments,
		'target_field_name': 'aspect_sentiments',
		'target': [('aspect_sentiments', aspect_sentiments)],
		'padding_field_name': 'padding',
		'examples': examples,
		'embeddings': (source_embedding, None),
		'dummy_input': Variable(torch.zeros((batch_size, 42), dtype=torch.long)),
		'baselines': {
			'germeval_baseline': 0.667,
			'germeval_best': 0.749
		}
		}