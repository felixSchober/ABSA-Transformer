import logging

import torch

from torch.autograd import *
import torchtext
from torchtext import data
from stop_words import get_stop_words

from data.torchtext.custom_fields import ReversibleField, ElmoField
from data.torchtext.custom_datasets import CustomGermEval2017Dataset
from data.data_loader import get_embedding, get_embedding_size

from misc.run_configuration import RunConfiguration

logger = logging.getLogger(__name__)

def preprocess_word(word: str) -> str:
	# TODO: Actual processing
	return word

def preprocess_relevance_word(word: str) -> int:
	if word == 'false':
		return 0
	return 1

def germeval2017_dataset(
					task:str,
					pretrained_vectors,
					hyperparameters: RunConfiguration,
					batch_size=80,
					root='./germeval2017',
					train_file='train_v1.4.tsv',
					validation_file='dev_v1.4.tsv',
					test_file=None,
					use_cuda=False,
					verbose=True):
	if hyperparameters.use_stop_words:
		stop_words = get_stop_words('de')
	else:
		stop_words = []

	# create an elmo field if we use elmo
	if hyperparameters.embedding_type == 'elmo':
		# contains the sentences               
		comment_field = ElmoField(
							hyperparameters.language,
							hyperparameters,
							batch_first=True,    # produce tensors with batch dimension first
							lower=True,
							fix_length=hyperparameters.clip_comments_to,
							sequential=True,
							use_vocab=True,
							init_token=None,
							eos_token=None,
							is_target=False,
							stop_words=stop_words,
							preprocessing=data.Pipeline(preprocess_word))
	else:
		comment_field = ReversibleField(
								batch_first=True,    # produce tensors with batch dimension first
								lower=True,
								fix_length=hyperparameters.clip_comments_to,
								sequential=True,
								use_vocab=True,
								init_token=None,
								eos_token=None,
								is_target=False,
								stop_words=stop_words,
								preprocessing=data.Pipeline(preprocess_word))

	relevant_field = data.Field(
							batch_first=True,
							is_target=True,
							sequential=False,
							use_vocab=False,
							unk_token=None,
							preprocessing=data.Pipeline(preprocess_relevance_word))
	
	id_field = ReversibleField(
							batch_first=True,
							is_target=False,
							sequential=False,
							use_vocab=True,
							unk_token=None)

	general_sentiment_field = ReversibleField(
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

	fields = [
		('id', id_field),                                   # link to comment eg: (http://twitter.com/lolloseb/statuses/718382187792478208)
		('comments', comment_field),                        # comment itself e.g. (@KuttnerSarah @DB_Bahn Hund = Fahrgast, Hund in Box = Gepäck.skurril, oder?)
		('relevance', relevant_field),                      # is comment relevant true/false
		('general_sentiments', general_sentiment_field),    # sentiment of comment (positive, negative, neutral)
		(None, None),                                       # apsect based sentiment e.g (Allgemein#Haupt:negative Sonstige_Unregelmässigkeiten#Haupt:negative Sonstige_Unregelmässigkeiten#Haupt:negative)
		('aspect_sentiments', aspect_sentiment_field),		# apsect sentiment field List of 20 aspects with positive, negative, neutral, n/a
		('padding', padding_field)                          # artificial field that we append to fill it with the padding information later to create the masks
			]

	train, val, test = CustomGermEval2017Dataset.splits(
							path=root,
							root='.data',
							train=train_file,
							validation=validation_file,
							test=test_file,
							separator='\t',
							fields=fields,
							verbose=verbose,
							hp=hyperparameters)

	# use updated fields
	fields = train.fields

	# build vocabularies
	if hyperparameters.embedding_type != 'elmo':
		comment_field.build_vocab(train.comments, val.comments, test.comments, vectors=[pretrained_vectors])
	else:
		comment_field.build_vocab(train.comments, val.comments, test.comments)

	general_sentiment_field.build_vocab(train.general_sentiments)
	padding_field.build_vocab(train.padding, val.comments, test.comments)
	aspect_sentiment_field.build_vocab(train.aspect_sentiments, val.aspect_sentiments, test.aspect_sentiments)
	id_field.build_vocab(train.id, val.id, test.id)

	# build aspect fields
	aspect_sentiment_fields = []
	for s_cat, f in train.aspect_sentiment_fields:
		f.build_vocab(train.__getattr__(s_cat), val.__getattr__(s_cat), test.__getattr__(s_cat))
		aspect_sentiment_fields.append(f)

	train_device = torch.device('cuda:0' if torch.cuda.is_available() and use_cuda else 'cpu')
	train_iter, val_iter, test_iter = data.BucketIterator.splits(
		(train, val, test), batch_size=batch_size, device=train_device, shuffle=True)

	# add embeddings
	embedding_size = get_embedding_size(comment_field, hyperparameters.embedding_type)
	source_embedding = get_embedding(comment_field.vocab, embedding_size, hyperparameters.embedding_type)

	examples = train.examples[0:3] + val.examples[0:3] + test.examples[0:3]

	return {
		'task': task,
		'split_length': (len(train), len(val), len(test)),
		'iters': (train_iter, val_iter, test_iter), 
		'vocabs': (comment_field.vocab, general_sentiment_field.vocab, aspect_sentiment_field.vocab),
		'fields': fields,
		'source_field_name': 'comments',
		'source_field': comment_field,
		'id_field': id_field,
		'aspect_sentiment_field': aspect_sentiment_field,
		'general_sentiment_field': general_sentiment_field,
		'target_field_name': 'general_sentiments',
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