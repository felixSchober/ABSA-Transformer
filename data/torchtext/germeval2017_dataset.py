
import os
import string
from typing import Dict, List, Tuple, Union
import pickle
from collections import defaultdict
import re
from unidecode import unidecode

import torchtext.data as data
from data.torchtext.custom_fields import ReversibleField
import torch.utils.data
from spellchecker import SpellChecker
from tqdm import tqdm
import spacy

from misc.utils import create_dir_if_necessary, check_if_file_exists
from data.torchtext.custom_datasets import *

class GermEval2017Dataset(Dataset):

	@staticmethod
	def sort_key(example):
		for attr in dir(example):
			if not callable(getattr(example, attr)) and \
					not attr.startswith("__"):
				return len(getattr(example, attr))
		return 0

	@classmethod
	def splits(cls, path=None, root='.data', train=None, validation=None,
			   test=None, **kwargs) -> Tuple[Dataset]:
		"""Create Dataset objects for multiple splits of a dataset.
		Arguments:
			path (str): Common prefix of the splits' file paths, or None to use
				the result of cls.download(root).
			root (str): Root dataset storage directory. Default is '.data'.
			train (str): Suffix to add to path for the train set, or None for no
				train set. Default is None.
			validation (str): Suffix to add to path for the validation set, or None
				for no validation set. Default is None.
			test (str): Suffix to add to path for the test set, or None for no test
				set. Default is None.
			Remaining keyword arguments: Passed to the constructor of the
				Dataset (sub)class being used.
		Returns:
			Tuple[Dataset]: Datasets for train, validation, and
			test splits in that order, if provided.
		"""
		if path is None:
			path = cls.download(root)

		train_data = None if train is None else cls(
			os.path.join(path, train), **kwargs)
		# make sure, we use exactly the same fields across all splits
		train_aspects = train_data.aspects

		val_data = None if validation is None else cls(
			os.path.join(path, validation), a_sentiment=train_aspects, **kwargs)

		test_data = None if test is None else cls(
			os.path.join(path, test), a_sentiment=train_aspects, **kwargs)

		return tuple(d for d in (train_data, val_data, test_data)
					 if d is not None)
	
	def __init__(self, path, fields, a_sentiment=[], separator='\t', **kwargs):
		self.aspect_sentiment_fields = []
		self.aspects = a_sentiment if len(a_sentiment) > 0 else []
		self.stats = defaultdict(get_stats_dd)
		self.na_labels = 0
		self.hp = None

		# first, try to load all models from cache
		filename = path.split("\\")[-1]
		examples, loaded_fields = self._try_load(filename.split(".")[0], fields)

		if not examples:
			examples, fields = self._load(path, filename, fields, a_sentiment, separator, **kwargs)
			self._save(filename.split(".")[0], examples)
		else:
			fields = loaded_fields
			
		super(GermEval2017Dataset, self).__init__(examples, tuple(fields))    

	def _load(self, path, filename, fields, a_sentiment=[], separator='\t', verbose=True, hp=None, **kwargs):
		examples = []
		self.hp = hp
		
		# remove punctuation
		punctuation_remover = str.maketrans('', '', string.punctuation + '…' + "“" + "„" + "‘")

		# In the end, those are the fields
		# The file has the aspect sentiment at the first aspect sentiment position
		# 0: link (id)
		# 1: Comment
		# 2: Is Relevant
		# 3: General Sentiment
		# 4: Apsect Specific sentiment List
		# 5: Padding Field
		# 6: aspect Sentiment 1/20
		# 7: aspect Sentiment 2/20
		# 8: aspect Sentiment 3/20
		# 9: aspect Sentiment 4/20
		# 10: aspect Sentiment 5/20
		# 11: aspect Sentiment 6/20
		# 12: aspect Sentiment 7/20
		# 13: aspect Sentiment 8/20
		# 14: aspect Sentiment 9/20
		# 15: aspect Sentiment 10/20
		# 16: aspect Sentiment 11/20
		# 17: aspect Sentiment 12/20
		# 18: aspect Sentiment 13/20
		# 19: aspect Sentiment 14/20
		# 20: aspect Sentiment 15/20
		# 21: aspect Sentiment 16/20
		# 22: aspect Sentiment 17/20
		# 23: aspect Sentiment 18/20
		# 24: aspect Sentiment 19/20
		# 25: aspect Sentiment 20/20

		if hp.use_spell_checkers:
			spell = self.initialize_spellchecker('de')
		else:
			spell = None

		with open(path, encoding="utf8") as input_file:
			aspect_sentiment_categories = set()
			aspect_sentiments: List[Dict[str, str]] = []

			raw_examples: List[List[Union[str, List[Dict[str, str]]]]] = []

			if verbose:
				iterator = tqdm(input_file, desc=f'Load {filename[0:7]}', leave=False)
			else:
				iterator = input_file

			for line in iterator:
				columns = []
				line = line.strip()
				if line == '':
					continue
				columns = line.split(separator)

				# aspect sentiment is missing
				if len(columns) == 4:
					columns.append('')
					columns.append(dict())
				else:
					# handle aspect sentiment which comes in a form of 
					# PART#Allgemein:negative PART#Allgemein:negative PART#Sicherheit:negative 

					# list of category - sentiment pair (Allgemein:negative)
					sentiments = columns[4]
					sentiments = sentiments.strip()
					sentiments = sentiments.split(' ')

					sentiment_dict = dict()

					for s in sentiments:
						category = ''
						sentiment = ''
						# remove #part
						s = s.split('#')

						if len(s) == 1:
							s = s[0]
							kv = s.split(':')
							category = kv[0]
							sentiment = kv[1]
						else:
							category = s[0]
							kv = s[1].split(':')
							sentiment = kv[1]

						sentiment_dict[category] = sentiment
						self.stats[category][sentiment] += 1
					 
					# add all new potential keys to set
					for s_category in sentiment_dict.keys():
						aspect_sentiment_categories.add(s_category)
					columns.append(sentiment_dict) 

				# remove punctuation and clean text
				comment = columns[1]

				# remove » and fix encoding issues
				comment = comment.replace('»', ' ')
				comment = comment.replace("ã¼", 'ü')
				comment = comment.replace("ã¤", "ä")
				comment = comment.replace("ø", "ö")
				comment = comment.replace("ű", "ü")
				comment = comment.replace("..", " ")


				# replace urls with regex
				comment = replace_urls_regex(comment)


				# remove non ascii characters with empty space
				# comment = re.sub(r'[^\x00-\x7f]',r' ', comment)
				#comment = unidecode(str(comment))
				
				# remove any non-word characters with empty space
				comment = re.sub(r'[^\w\säöüß]', r' ', comment)

				comment = comment.split(' ')

				# remove all empty entries
				comment = [w for w in comment if w.strip() != '']

				if hp.harmonize_bahn:
					comment = harmonize_bahn_names(comment)

				if hp.replace_url_tokens:
					comment = replace_urls(comment)


				if hp.use_spell_checkers:
					comment = self.fix_spellings(comment, spell, 'de')

				comment = ' '.join(comment)
				#comment = comment.translate(punctuation_remover)

				columns[1] = comment

				# comment is not relevant
				if columns[2] == 'false':
					# set all aspects to n/a
					pass
					
				# add aspect sentiment field
				columns.append('')

				# add padding field
				columns.append('')
				raw_examples.append(columns)

		# process the aspect sentiment
		if len(self.aspects) == 0:
			aspect_sentiment_categories.add('QR-Code')
			self.aspects = list(aspect_sentiment_categories)

			# make sure the list is sorted. Otherwise we'll have a different
			# order every time and can not transfer models
			self.aspects = sorted(self.aspects)

			# construct the fields
			fields = self._construct_fields(fields)

		for raw_example in raw_examples:
			# go through each aspect sentiment and add it at the corresponding position
			ss = ['n/a'] * len(self.aspects)
			nas = len(self.aspects)
			for s_category, s in raw_example[-3].items():
				pos = self.aspects.index(s_category)
				ss[pos] = s
				nas -= 1

			self.na_labels += nas
			raw_example[6] = ss
			
			# construct example and add it
			example = raw_example[0:5] + [raw_example[6]] + [raw_example[7]] + ss
			examples.append(data.Example.fromlist(example, tuple(fields)))

		# clip comments
		for example in examples:
			comment_length: int = len(example.comments)
			if comment_length > hp.clip_comments_to:
				example.comments = example.comments[0:hp.clip_comments_to]
				comment_length = hp.clip_comments_to

			example.padding = ['0'] * comment_length
		return examples, fields
		
	def _construct_fields(self, fields):
		for s_cat in self.aspects:

				f = ReversibleField(
								batch_first=True,
								is_target=True,
								sequential=False,
								init_token=None,
								eos_token=None,
								unk_token=None,
								use_vocab=True)
				self.aspect_sentiment_fields.append((s_cat, f))
				fields.append((s_cat, f))
		return fields

	def _try_load(self, name, fields):
		path = os.path.join(os.getcwd(), 'data', 'data', 'cache')
		create_dir_if_necessary(path)
		samples_path = os.path.join(path, name + "2.pkl")
		aspects_path = os.path.join(path, name + "_2aspects.pkl")

		if not check_if_file_exists(samples_path) or not check_if_file_exists(aspects_path):
			return [], None

		with open(samples_path, 'rb') as f:
			examples = pickle.load(f)

		with open(aspects_path, 'rb') as f:
			self.aspects = pickle.load(f)

		# get all fields
		fields = self._construct_fields(fields)
		return examples, fields

	def _save(self, name, samples):
		path = os.path.join(os.getcwd(), 'data', 'data', 'cache')
		create_dir_if_necessary(path)
		samples_path = os.path.join(path, name + ".pkl")
		aspects_path = os.path.join(path, name + "_aspects.pkl")

		# print(f'Trying to save loaded dataset to {samples_path}.')
		with open(samples_path, 'wb') as f:
			pickle.dump(samples, f)
			# print(f'Model {name} successfully saved.')

		with open(aspects_path, "wb") as f:
			pickle.dump(self.aspects, f)


def harmonize_bahn_names(text_tokens: List[str]) -> List[str]:
	bahn_syn = [
		'db',
		'deutschebahn',
		"db_bahn",
		"bahn.de",
		"@db_bahn",
		"#db",
		"@db",
		"#db_bahn",
		"@bahn",
		"#bahn",
		"@dbbahn",
		"#dbbahn",
		"#dbbahn"
		"www.bahn.de",
		"dbbahn"
	]
	result = []
	for token in text_tokens:
		if token.lower() in bahn_syn:
			result.append('db')
		else:
			result.append(token)
	return result
