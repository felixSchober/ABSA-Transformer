import torchtext.data as data
import re
import io
import os
import pickle
import zipfile
import tarfile
import gzip
import shutil
from functools import partial
import string
from typing import Dict, List, Tuple, Union

import torch.utils.data

from torchtext.data.utils import RandomShuffler
from torchtext.utils import download_from_url, unicode_csv_reader
from data.torchtext.custom_fields import ReversibleField
#from tqdm.autonotebook import tqdm
from tqdm import tqdm
import spacy
from spellchecker import SpellChecker
from misc.utils import create_dir_if_necessary, check_if_file_exists


class Dataset(torch.utils.data.Dataset):
	"""Defines a dataset composed of Examples along with its Fields.
	Attributes:
		sort_key (callable): A key to use for sorting dataset examples for batching
			together examples with similar lengths to minimize padding.
		examples (list(Example)): The examples in this dataset.
		fields (dict[str, Field]): Contains the name of each column or field, together
			with the corresponding Field object. Two fields with the same Field object
			will have a shared vocabulary.
	"""
	sort_key = None

	def __init__(self, examples, fields, filter_pred=None):
		"""Create a dataset from a list of Examples and Fields.
		Arguments:
			examples: List of Examples.
			fields (List(tuple(str, Field))): The Fields to use in this tuple. The
				string is a field name, and the Field is the associated field.
			filter_pred (callable or None): Use only examples for which
				filter_pred(example) is True, or use all examples if None.
				Default is None.
		"""
		if filter_pred is not None:
			make_list = isinstance(examples, list)
			examples = filter(filter_pred, examples)
			if make_list:
				examples = list(examples)
		self.examples = examples
		self.fields = dict(fields)
		# Unpack field tuples
		for n, f in list(self.fields.items()):
			if isinstance(n, tuple):
				self.fields.update(zip(n, f))
				del self.fields[n]

	@classmethod
	def splits(cls, path=None, root='.data', train=None, validation=None,
			   test=None, **kwargs):
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
		train_fields = train_data.fields

		val_data = None if validation is None else cls(
			os.path.join(path, validation), fields=train_data.fields, **kwargs)

		test_data = None if test is None else cls(
			os.path.join(path, test), fields=train_data.fields, **kwargs)

		return tuple(d for d in (train_data, val_data, test_data)
					 if d is not None)

	def split(self, split_ratio=0.7, stratified=False, strata_field='label',
			  random_state=None):
		"""Create train-test(-valid?) splits from the instance's examples.
		Arguments:
			split_ratio (float or List of floats): a number [0, 1] denoting the amount
				of data to be used for the training split (rest is used for validation),
				or a list of numbers denoting the relative sizes of train, test and valid
				splits respectively. If the relative size for valid is missing, only the
				train-test split is returned. Default is 0.7 (for the train set).
			stratified (bool): whether the sampling should be stratified.
				Default is False.
			strata_field (str): name of the examples Field stratified over.
				Default is 'label' for the conventional label field.
			random_state (tuple): the random seed used for shuffling.
				A return value of `random.getstate()`.
		Returns:
			Tuple[Dataset]: Datasets for train, validation, and
			test splits in that order, if the splits are provided.
		"""
		train_ratio, test_ratio, val_ratio = check_split_ratio(split_ratio)

		# For the permutations
		rnd = RandomShuffler(random_state)
		if not stratified:
			train_data, test_data, val_data = rationed_split(self.examples, train_ratio,
															 test_ratio, val_ratio, rnd)
		else:
			if strata_field not in self.fields:
				raise ValueError("Invalid field name for strata_field {}"
								 .format(strata_field))
			strata = stratify(self.examples, strata_field)
			train_data, test_data, val_data = [], [], []
			for group in strata:
				# Stratify each group and add together the indices.
				group_train, group_test, group_val = rationed_split(group, train_ratio,
																	test_ratio, val_ratio,
																	rnd)
				train_data += group_train
				test_data += group_test
				val_data += group_val

		splits = tuple(Dataset(d, self.fields)
					   for d in (train_data, val_data, test_data) if d)

		# In case the parent sort key isn't none
		if self.sort_key:
			for subset in splits:
				subset.sort_key = self.sort_key
		return splits

	def __getitem__(self, i):
		return self.examples[i]

	def __len__(self):
		try:
			return len(self.examples)
		except TypeError:
			return 2**32

	def __iter__(self):
		for x in self.examples:
			yield x

	def __getattr__(self, attr):
		if attr in self.fields:
			for x in self.examples:
				yield getattr(x, attr)

	@classmethod
	def download(cls, root, check=None):
		"""Download and unzip an online archive (.zip, .gz, or .tgz).
		Arguments:
			root (str): Folder to download data to.
			check (str or None): Folder whose existence indicates
				that the dataset has already been downloaded, or
				None to check the existence of root/{cls.name}.
		Returns:
			str: Path to extracted dataset.
		"""
		path = os.path.join(root, cls.name)
		check = path if check is None else check
		if not os.path.isdir(check):
			for url in cls.urls:
				if isinstance(url, tuple):
					url, filename = url
				else:
					filename = os.path.basename(url)
				zpath = os.path.join(path, filename)
				if not os.path.isfile(zpath):
					if not os.path.exists(os.path.dirname(zpath)):
						os.makedirs(os.path.dirname(zpath))
					print('downloading {}'.format(filename))
					download_from_url(url, zpath)
				zroot, ext = os.path.splitext(zpath)
				_, ext_inner = os.path.splitext(zroot)
				if ext == '.zip':
					with zipfile.ZipFile(zpath, 'r') as zfile:
						print('extracting')
						zfile.extractall(path)
				# tarfile cannot handle bare .gz files
				elif ext == '.tgz' or ext == '.gz' and ext_inner == '.tar':
					with tarfile.open(zpath, 'r:gz') as tar:
						dirs = [member for member in tar.getmembers()]
						tar.extractall(path=path, members=dirs)
				elif ext == '.gz':
					with gzip.open(zpath, 'rb') as gz:
						with open(zroot, 'wb') as uncompressed:
							shutil.copyfileobj(gz, uncompressed)

		return os.path.join(path, cls.dirname)

	def filter_examples(self, field_names):
		"""Remove unknown words from dataset examples with respect to given field.
		Arguments:
			field_names (list(str)): Within example only the parts with field names in
				field_names will have their unknown words deleted.
		"""
		for i, example in enumerate(self.examples):
			for field_name in field_names:
				vocab = set(self.fields[field_name].vocab.stoi)
				text = getattr(example, field_name)
				example_part = [word for word in text if word in vocab]
				setattr(example, field_name, example_part)
			self.examples[i] = example

class CustomSequenceTaggingDataSet(Dataset):
	"""Defines a dataset for sequence tagging. Examples in this dataset
	contain paired lists -- paired list of words and tags.

	For example, in the case of part-of-speech tagging, an example is of the
	form
	[I, love, PyTorch, .] paired with [PRON, VERB, PROPN, PUNCT]

	See torchtext/test/sequence_tagging.py on how to use this class.
	"""

	@staticmethod
	def sort_key(example):
		for attr in dir(example):
			if not callable(getattr(example, attr)) and \
					not attr.startswith("__"):
				return len(getattr(example, attr))
		return 0

	def __init__(self, path, fields, separator="\t", **kwargs):
		examples = []
		columns = []

		with open(path) as input_file:
			for line in input_file:
				if line.startswith("-DOCSTART-"):
					continue
				
				line = line.strip()
				if line == "":
					if columns:
						# copy first column as a complete sentence that is not tokenized
						#sentence = columns[0].copy()
						#columns.append(sentence)
						examples.append(data.Example.fromlist(columns, fields))
					columns = []
				else:
					for i, column in enumerate(line.split(separator)):
						if len(columns) < i + 1:
							columns.append([])
						columns[i].append(column)

			if columns:
				examples.append(data.Example.fromlist(columns, fields))
		super(CustomSequenceTaggingDataSet, self).__init__(examples, fields,
													 **kwargs)

class CustomGermEval2017Dataset(Dataset):

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

		# first, try to load all models from cache
		filename = path.split("\\")[-1]
		examples, loaded_fields = self._try_load(filename.split(".")[0], fields)

		if not examples:
			examples, fields = self._load(path, filename, fields, a_sentiment, separator, **kwargs)
			self._save(filename.split(".")[0], examples)
		else:
			fields = loaded_fields
			
		super(CustomGermEval2017Dataset, self).__init__(examples, tuple(fields))    

	def _load(self, path, filename, fields, a_sentiment=[], separator='\t', verbose=True, hp=None, **kwargs):
		examples = []
		
		# remove punctuation
		punctuation_remover = str.maketrans('', '', string.punctuation + '…' + "“" + "–" + "„")

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
			spell = SpellChecker(language='de')  # German dictionary
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
					 
					# add all new potential keys to set
					for s_category in sentiment_dict.keys():
						aspect_sentiment_categories.add(s_category)
					columns.append(sentiment_dict) 

				# remove punctuation and clean text
				comment = columns[1]

				comment = comment.split(' ')
				if hp.harmonize_bahn:
					comment = harmonize_bahn_names(comment)

				if hp.replace_url_tokens:
					comment = remove_websites(comment)

				comment = ' '.join(comment)

				if hp.use_spell_checkers:
					comment = text_cleaner(comment, 'de', spell)
				comment = comment.translate(punctuation_remover)

				columns[1] = comment
				# comment is not relevant
				if columns[2] == 'false':
					# skip for now
					# pass
					continue
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
			for s_category, s in raw_example[-3].items():
				pos = self.aspects.index(s_category)
				ss[pos] = s

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

# mapping for organic dataset aspects
od_entity_mapping = {
	'g': 'organic general',
	'p': 'organic products',
	'f': 'organic farmers',
	'c': 'organic companies',
	'cg': 'conventional general',
	'cp': 'conventional products',
	'cf': 'conventional farming',
	'cc': 'conventional companies',
	'gg': 'GMOs genetic engineering general'
}

od_attribute_mapping = {
	'g': 'general',
	'p': 'price',
	't': 'taste',
	'q': 'Nutr. quality & freshness',
	's': 'safety',
	'h': 'healthiness',
	'c': 'chemicals pesticides',
	'll': 'label',
	'or': 'origin source',
	'l': 'local',
	'av': 'availability',
	'a': 'animal welfare',
	'pp': 'productivity'
}

od_sentiment_mapping = {
	'0': 'neutral',
	'p': 'positive',
	'n': 'negative'
}

def get_all_mapping():
	result = {}
	for entity_key, entity in od_entity_mapping:
		for attribute_key, attribute in od_attribute_mapping:
			compound_key = f'{entity_key}-{attribute_key}'
			result[compound_key] = f'{entity}: {attribute}'
	return result

class CustomSentenceWiseBioDataset(Dataset):

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

		# lines for splits
		lengths = (8918, 786, 738)

		train_data = None if train is None else cls(
			os.path.join(path, train), length=lengths[0], **kwargs)
		# make sure, we use exactly the same fields across all splits
		train_aspects = train_data.aspects

		val_data = None if validation is None else cls(
			os.path.join(path, validation), a_sentiment=train_aspects, length=lengths[1], **kwargs)

		test_data = None if test is None else cls(
			os.path.join(path, test), a_sentiment=train_aspects, length=lengths[2], **kwargs)

		return tuple(d for d in (train_data, val_data, test_data)
					 if d is not None)
	
	def __init__(self, path, fields, a_sentiment=[], separator='\t', task=None, **kwargs):
		self.aspect_sentiment_fields = []
		self.aspects = a_sentiment if len(a_sentiment) > 0 else []
		self.dataset_name = 'organic2019Sentences'

		# first, try to load all models from cache
		_, filename = os.path.split(path)
		filename = f'{filename.split(".")[0]}_{task}'

		examples, loaded_fields = self._try_load(filename, fields)

		if not examples:
			examples, fields = self._load(path, filename, fields, a_sentiment, separator, task=task, **kwargs)
			self._save(filename, examples)
		else:
			fields = loaded_fields
			
		super(CustomSentenceWiseBioDataset, self).__init__(examples, tuple(fields))    

	def _load(self, path, filename, fields, a_sentiment=[], separator='|', verbose=True, hp=None, task=None, length=None, **kwargs):
		examples = []
		
		# remove punctuation
		punctuation_remover = str.maketrans('', '', string.punctuation + '…' + "“" + "–" + "„")

		# 0: Sequence number
		# 1: Index
		# 2: Author_Id
		# 3: Comment number
		# 4: Sentence number
		# 5: Domain Relevance
		# 6: Sentiment
		# 7: Entity
		# 8: Attribute
		# 9: Sentence
		# 10: Source File
		# 11: Apsect Specific sentiment List
		# 12: Padding Field
		# 13+: aspect Sentiment 1/n

		if hp.use_spell_checkers:
			spell = initialize_spellchecker(hp.language)

		else:
			spell = None

		if task == 'all':
			aspect_example_index = -1
			mapping = get_all_mapping()
		elif task == 'entities':
			aspect_example_index = -5
			mapping = od_entity_mapping
		elif task == 'attributes':
			mapping = od_attribute_mapping
			aspect_example_index = - 4

		with open(path, 'rb') as input_file:
			aspect_sentiment_categories = set()
			aspect_sentiments: List[Dict[str, str]] = []

			raw_examples: List[List[Union[str, List[Dict[str, str]]]]] = []

			if verbose:
				iterator = tqdm(input_file, desc=f'Load {filename[0:7]}', leave=False, total=length)
			else:
				iterator = input_file

			last_sentence_number = None
			last_comment_number = None
			last_sample = None

			# skip the first line
			skip_line = True

			for line in iterator:
				line = line.decode(errors='ignore')
				columns = []
				line = line.strip()
				if skip_line or line == '':
					skip_line = False
					continue
				columns = line.split(separator)

				if columns[-1] == 'Entity-Attribute':
					continue

				# comment is not relevant
				if columns[6] == '0' or columns[-1] == '':
					# skip for now
					# pass
					continue

				# aspect sentiment is missing
				if len(columns) == 12:
					columns.append('')
					columns.append(dict())
					last_sample = columns
				else:
					# based on aspect task select columns
					aspect_category = columns[aspect_example_index].strip()

					# use mapping to get a more human readable name
					aspect_category = mapping[aspect_category]
					
					aspect_sentiment = od_sentiment_mapping[columns[7].strip()]		

					crnt_sentence_number = columns[5]
					crnt_comment_number = columns[4]
					# if last_sentence_number and last_comment are set and equal this means we need to add to the sentiment dict
					# otherwise we add the last sample and move on
					
					# case 1: not set 
					#	-> first comment
					if last_sentence_number is None or last_comment_number is None:
						last_sentence_number = crnt_sentence_number
						last_comment_number = crnt_comment_number
						comment_sentiment_dict = dict()
						last_sample = columns


					# case 2: last and current do not numbers match
					# new sample -> add to new dict
					elif last_sentence_number != crnt_sentence_number or last_comment_number != crnt_comment_number:
						# add last sample
						# add all new potential keys to set
						for s_category in comment_sentiment_dict.keys():
							aspect_sentiment_categories.add(s_category)
						last_sample.append(comment_sentiment_dict) 

						comment_sentiment_dict = dict()
						last_sentence_number = crnt_sentence_number
						last_comment_number = crnt_comment_number
						
						raw_examples.append(last_sample)
						last_sample = columns

					# case 3: last and current match
					# 		-> add to last sample
					elif last_sentence_number == crnt_sentence_number and last_comment_number == crnt_comment_number:
						comment_sentiment_dict[aspect_category] = aspect_sentiment
						continue	

					comment_sentiment_dict[aspect_category] = aspect_sentiment
					
								

				# remove punctuation and clean text
				comment = last_sample[-3]
				comment = comment.translate(punctuation_remover)

				# remove non ascii characters with empty space
				comment = re.sub(r'[^\x00-\x7f]',r' ', comment)
				
				if hp.language == 'en':
					comment = en_contraction_removal(comment)

				comment = comment.split(' ')

				if hp.replace_url_tokens:
					comment = remove_websites(comment)

				if hp.use_spell_checkers:
					comment = fix_spellings(comment, spell)

				comment = ' '.join(comment)
				if hp.use_text_cleaner:
					comment = text_cleaner(comment, hp.language, spell)

				last_sample[-3] = comment
				
				# add aspect sentiment field
				last_sample.append('')

				# add padding field
				last_sample.append('')

		# process the aspect sentiment
		if len(self.aspects) == 0:
			#aspect_sentiment_categories.add('QR-Code')
			self.aspects = list(aspect_sentiment_categories)

			# construct the fields
			fields = self._construct_fields(fields)

		for raw_example in raw_examples:
			# go through each aspect sentiment and add it at the corresponding position
			ss = ['n/a'] * len(self.aspects)
			for s_category, s in raw_example[-1].items():
				pos = self.aspects.index(s_category)
				ss[pos] = s

			raw_example[6] = ss


			# 0: Sequence number
			# 1: Index
			# 2: Author_Id
			# 3: Comment number
			# 4: Sentence number
			# 5: Domain Relevance
			# 6: Sentiment
			# 7: Sentence
			# 8: Padding
			# 9: Source File
			# 10+: aspect Sentiment 1/n
			
			# construct example and add it
			example = raw_example[0:6] + [raw_example[6]] + [raw_example[10], '', ''] + ss
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
		path = os.path.join(os.getcwd(), 'data', 'data', 'cache', self.dataset_name)
		create_dir_if_necessary(path)
		samples_path = os.path.join(path, name + ".pkl")
		aspects_path = os.path.join(path, name + "_aspects.pkl")

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
		path = os.path.join(os.getcwd(), 'data', 'cache', self.dataset_name)
		create_dir_if_necessary(path)
		samples_path = os.path.join(path, name + ".pkl")
		aspects_path = os.path.join(path, name + "_aspects.pkl")

		# print(f'Trying to save loaded dataset to {samples_path}.')
		with open(samples_path, 'wb') as f:
			pickle.dump(samples, f)
			# print(f'Model {name} successfully saved.')

		with open(aspects_path, "wb") as f:
			pickle.dump(self.aspects, f) 

class CustomCommentWiseBioDataset(Dataset):
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

		# lines for splits
		lengths = (8918, 786, 738)

		train_data = None if train is None else cls(
			os.path.join(path, train), length=lengths[0], **kwargs)
		# make sure, we use exactly the same fields across all splits
		train_aspects = train_data.aspects

		val_data = None if validation is None else cls(
			os.path.join(path, validation), a_sentiment=train_aspects, length=lengths[1], **kwargs)

		test_data = None if test is None else cls(
			os.path.join(path, test), a_sentiment=train_aspects, length=lengths[2], **kwargs)

		return tuple(d for d in (train_data, val_data, test_data)
					 if d is not None)
	
	def __init__(self, path, fields, a_sentiment=[], separator='\t', task=None, **kwargs):
		self.aspect_sentiment_fields = []
		self.aspects = a_sentiment if len(a_sentiment) > 0 else []
		self.dataset_name = 'organic2019Comments'

		# first, try to load all models from cache
		_, filename = os.path.split(path)
		filename = f'{filename.split(".")[0]}_{task}'

		examples, loaded_fields = self._try_load(filename, fields)

		if not examples:
			examples, fields = self._load(path, filename, fields, a_sentiment, separator, task=task, **kwargs)
			self._save(filename, examples)
		else:
			fields = loaded_fields
			
		super(CustomCommentWiseBioDataset, self).__init__(examples, tuple(fields))    

	def _load(self, path, filename, fields, a_sentiment=[], separator='|', verbose=True, hp=None, task=None, length=None, **kwargs):
		examples = []
		
		# remove punctuation
		punctuation_remover = str.maketrans('', '', string.punctuation + '…' + "“" + "–" + "„")

		# 0: Sequence number
		# 1: Index
		# 2: Author_Id
		# 3: Comment number
		# 4: Sentence number
		# 5: Domain Relevance
		# 6: Sentiment
		# 7: Entity
		# 8: Attribute
		# 9: Sentence
		# 10: Source File
		# 11: Apsect Specific sentiment List
		# 12: Padding Field
		# 13+: aspect Sentiment 1/n

		if hp.use_spell_checkers:
			spell = initialize_spellchecker(hp.language)

		else:
			spell = None

		if task == 'all_combine':
			aspect_example_index = -1
			mapping = get_all_mapping()
		elif task == 'entities_combine':
			aspect_example_index = -5
			mapping = od_entity_mapping
		elif task == 'attributes_combine':
			mapping = od_attribute_mapping
			aspect_example_index = - 4

		comments = {}
		with open(path, 'rb') as input_file:
			aspect_sentiment_categories = set()
			aspect_sentiments: List[Dict[str, str]] = []

			raw_examples: List[List[Union[str, List[Dict[str, str]]]]] = []

			if verbose:
				iterator = tqdm(input_file, desc=f'Load {filename[0:7]}', leave=False, total=length)
			else:
				iterator = input_file

			last_sentence_number = None
			last_comment_number = None
			last_sample = None

			# skip the first line
			skip_line = True

			for line in iterator:
				line = line.decode(errors='ignore')
				columns = []
				line = line.strip()
				if skip_line or line == '':
					skip_line = False
					continue
				columns = line.split(separator)

				if columns[-1] == 'Entity-Attribute':
					continue

				# comment is not relevant
				if columns[6] == '0' or columns[-1] == '':
					# skip for now
					# pass
					continue

				# aspect sentiment is missing
				if len(columns) == 12:
					columns.append('')
					columns.append(dict())
					last_sample = columns
				else:
					# based on aspect task select columns
					aspect_category = columns[aspect_example_index].strip()

					# use mapping to get a more human readable name
					aspect_category = mapping[aspect_category]
					
					aspect_sentiment = od_sentiment_mapping[columns[7].strip()]		

					crnt_sentence_number = columns[5]
					crnt_comment_number = columns[4]
					# if last_sentence_number and last_comment are set and equal this means we need to add to the sentiment dict
					# otherwise we add the last sample and move on
					
					# case 1: not set 
					#	-> first comment
					if last_sentence_number is None or last_comment_number is None:
						last_sentence_number = crnt_sentence_number
						last_comment_number = crnt_comment_number
						comment_sentiment_dict = dict()
						last_sample = columns


					# case 2: last and current do not numbers match
					# new sample -> add to new dict
					elif last_sentence_number != crnt_sentence_number or last_comment_number != crnt_comment_number:
						# add last sample
						# add all new potential keys to set
						for s_category in comment_sentiment_dict.keys():
							aspect_sentiment_categories.add(s_category)
						last_sample.append(comment_sentiment_dict) 

						comment_sentiment_dict = dict()
						last_sentence_number = crnt_sentence_number
						last_comment_number = crnt_comment_number

						if crnt_comment_number not in comments:
							comments[crnt_comment_number] = []

						comments[crnt_comment_number].append(last_sample)
						last_sample = columns

					# case 3: last and current match
					# 		-> add to last sample
					elif last_sentence_number == crnt_sentence_number and last_comment_number == crnt_comment_number:
						comment_sentiment_dict[aspect_category] = aspect_sentiment
						continue	

					comment_sentiment_dict[aspect_category] = aspect_sentiment
					
								

				# remove punctuation and clean text
				comment = last_sample[-3]
				comment = comment.translate(punctuation_remover)

				# remove non ascii characters with empty space
				comment = re.sub(r'[^\x00-\x7f]',r' ', comment)
				
				if hp.language == 'en':
					comment = en_contraction_removal(comment)

				comment = comment.split(' ')

				if hp.replace_url_tokens:
					comment = remove_websites(comment)

				if hp.use_spell_checkers:
					comment = fix_spellings(comment, spell)

				comment = ' '.join(comment)
				if hp.use_text_cleaner:
					comment = text_cleaner(comment, hp.language, spell)

				last_sample[-3] = comment
				
				# add aspect sentiment field
				last_sample.append('')

				# add padding field
				last_sample.append('')

		# first, find out the longest comment length (most sentences)
		#max_sentences = max([len(c) for c in comments.items()])

		max_dual_sentence_length = hp.clip_comments_to // 2
		for comment_sentences in comments.values():

			# 1st sentence per comment does not have a previous sentence
			s = comment_sentences[0]
			raw_examples.append(s)
			sentences = [s[-6] for s in comment_sentences]
			for i in range(len(comment_sentences) - 1):
				first_comment_text = sentences[i]
				second_comment_text = sentences[i+1]

				# prepend this text to the next comment and clip both comments
				comment_sentences[i+1][-6] = f'{first_comment_text[:max_dual_sentence_length]} {second_comment_text[:max_dual_sentence_length]}'
				raw_examples.append(comment_sentences[i+1])

		# process the aspect sentiment
		if len(self.aspects) == 0:
			self.aspects = list(aspect_sentiment_categories)

			# construct the fields
			fields = self._construct_fields(fields)

		for raw_example in raw_examples:
			# go through each aspect sentiment and add it at the corresponding position
			ss = ['n/a'] * len(self.aspects)
			for s_category, s in raw_example[-1].items():
				pos = self.aspects.index(s_category)
				ss[pos] = s

			raw_example[6] = ss


			# 0: Sequence number
			# 1: Index
			# 2: Author_Id
			# 3: Comment number
			# 4: Sentence number
			# 5: Domain Relevance
			# 6: Sentiment
			# 7: Sentence
			# 8: Padding
			# 9: Source File
			# 10+: aspect Sentiment 1/n
			
			# construct example and add it
			example = raw_example[0:6] + [raw_example[6]] + [raw_example[10], '', ''] + ss
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
		path = os.path.join(os.getcwd(), 'data', 'data', 'cache', self.dataset_name)
		create_dir_if_necessary(path)
		samples_path = os.path.join(path, name + ".pkl")
		aspects_path = os.path.join(path, name + "_aspects.pkl")

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
		path = os.path.join(os.getcwd(), 'data', 'cache', self.dataset_name)
		create_dir_if_necessary(path)
		samples_path = os.path.join(path, name + ".pkl")
		aspects_path = os.path.join(path, name + "_aspects.pkl")

		# print(f'Trying to save loaded dataset to {samples_path}.')
		with open(samples_path, 'wb') as f:
			pickle.dump(samples, f)
			# print(f'Model {name} successfully saved.')

		with open(aspects_path, "wb") as f:
			pickle.dump(self.aspects, f) 

def check_split_ratio(split_ratio):
	"""Check that the split ratio argument is not malformed"""
	valid_ratio = 0.
	if isinstance(split_ratio, float):
		# Only the train set relative ratio is provided
		# Assert in bounds, validation size is zero
		assert 0. < split_ratio < 1., (
			"Split ratio {} not between 0 and 1".format(split_ratio))

		test_ratio = 1. - split_ratio
		return (split_ratio, test_ratio, valid_ratio)
	elif isinstance(split_ratio, list):
		# A list of relative ratios is provided
		length = len(split_ratio)
		assert length == 2 or length == 3, (
			"Length of split ratio list should be 2 or 3, got {}".format(split_ratio))

		# Normalize if necessary
		ratio_sum = sum(split_ratio)
		if not ratio_sum == 1.:
			split_ratio = [float(ratio) / ratio_sum for ratio in split_ratio]

		if length == 2:
			return tuple(split_ratio + [valid_ratio])
		return tuple(split_ratio)
	else:
		raise ValueError('Split ratio must be float or a list, got {}'
						 .format(type(split_ratio)))

def stratify(examples, strata_field):
	# The field has to be hashable otherwise this doesn't work
	# There's two iterations over the whole dataset here, which can be
	# reduced to just one if a dedicated method for stratified splitting is used
	unique_strata = set(getattr(example, strata_field) for example in examples)
	strata_maps = {s: [] for s in unique_strata}
	for example in examples:
		strata_maps[getattr(example, strata_field)].append(example)
	return list(strata_maps.values())

def rationed_split(examples, train_ratio, test_ratio, val_ratio, rnd):
	# Create a random permutation of examples, then split them
	# by ratio x length slices for each of the train/test/dev? splits
	N = len(examples)
	randperm = rnd(range(N))
	train_len = int(round(train_ratio * N))

	# Due to possible rounding problems
	if not val_ratio:
		test_len = N - train_len
	else:
		test_len = int(round(test_ratio * N))

	indices = (randperm[:train_len],  # Train
			   randperm[train_len:train_len + test_len],  # Test
			   randperm[train_len + test_len:])  # Validation

	# There's a possibly empty list for the validation set
	data = tuple([examples[i] for i in index] for index in indices)

	return data

def text_cleaner(text: str, language: str, spellChecker):

	if language == 'en':
		text = en_contraction_removal(text)
	spacy_nlp = spacy.load(language)
	parsed = spacy_nlp(text)
	final_tokens = []
	for t in parsed:

		if t.is_punct or t.is_space or t.like_num or t.like_url or str(t).startswith('@'):
			continue

		if spellChecker is not None:
			# test if word is spelled correctly
			pass
		
		if t.lemma_ == '-PRON-':
			final_tokens.append(str(t))
		else:
			sc_removed = re.sub("[^a-zA-Zäöüß]", '', str(t.lemma_))
			if len(sc_removed) > 1:
				final_tokens.append(sc_removed)
	joined = ' '.join(final_tokens)
	spell_corrected = re.sub(r'(.)\1+', r'\1\1', joined)
	return spell_corrected

def en_contraction_removal(text: str) -> str:
	apostrophe_handled = re.sub("’", "'", text)
	# from https://gist.githubusercontent.com/tthustla/74e99a00541264e93c3bee8b2b49e6d8/raw/599100471e8127d6efad446717dc951a10b69777/yatwapart1_01.py
	contraction_mapping = {
					"youre": "you are",
					"youll": "you will",
					"theyre": "they are", "theyll": "they will",
					"weve": "we have",
					"shouldnt": "should not",
					"dont": "do not",
					"doesnt": "does not", "doesn": "does not",
					"didnt": "did not",
					"wasn": "was not",
					"arent": "are not", "aren": "are not",
					"aint": "is not", "isnt": "is not", "isn": "is not",
					"wouldnt": "would not", "wouldn": "would not",
					"ain't": "is not", "aren't": "are not","can't": "cannot", 
				   "can't've": "cannot have", "'cause": "because", "could've": "could have", 
				   "couldn't": "could not", "couldn't've": "could not have","didn't": "did not", 
				   "doesn't": "does not", "don't": "do not", "hadn't": "had not", 
				   "hadn't've": "had not have", "hasn't": "has not", "haven't": "have not", 
				   "he'd": "he would", "he'd've": "he would have", "he'll": "he will", 
				   "he'll've": "he will have", "he's": "he is", "how'd": "how did", 
				   "how'd'y": "how do you", "how'll": "how will", "how's": "how is", 
				   "I'd": "I would", "I'd've": "I would have", "I'll": "I will", 
				   "I'll've": "I will have","I'm": "I am", "I've": "I have", 
				   "i'd": "i would", "i'd've": "i would have", "i'll": "i will", 
				   "i'll've": "i will have","i'm": "i am", "i've": "i have", 
				   "isn't": "is not", "it'd": "it would", "it'd've": "it would have", 
				   "it'll": "it will", "it'll've": "it will have","it's": "it is", 
				   "let's": "let us", "ma'am": "madam", "mayn't": "may not", 
				   "might've": "might have","mightn't": "might not","mightn't've": "might not have", 
				   "must've": "must have", "mustn't": "must not", "mustn't've": "must not have", 
				   "needn't": "need not", "needn't've": "need not have","o'clock": "of the clock", 
				   "oughtn't": "ought not", "oughtn't've": "ought not have", "shan't": "shall not",
				   "sha'n't": "shall not", "shan't've": "shall not have", "she'd": "she would", 
				   "she'd've": "she would have", "she'll": "she will", "she'll've": "she will have", 
				   "she's": "she is", "should've": "should have", "shouldn't": "should not", 
				   "shouldn't've": "should not have", "so've": "so have","so's": "so as", 
				   "this's": "this is",
				   "that'd": "that would", "that'd've": "that would have","that's": "that is", 
				   "there'd": "there would", "there'd've": "there would have","there's": "there is", 
					   "here's": "here is",
				   "they'd": "they would", "they'd've": "they would have", "they'll": "they will", 
				   "they'll've": "they will have", "they're": "they are", "they've": "they have", 
				   "to've": "to have", "wasn't": "was not", "we'd": "we would", 
				   "we'd've": "we would have", "we'll": "we will", "we'll've": "we will have", 
				   "we're": "we are", "we've": "we have", "weren't": "were not", 
				   "what'll": "what will", "what'll've": "what will have", "what're": "what are", 
				   "what's": "what is", "what've": "what have", "when's": "when is", 
				   "when've": "when have", "where'd": "where did", "where's": "where is", 
				   "where've": "where have", "who'll": "who will", "who'll've": "who will have", 
				   "who's": "who is", "who've": "who have", "why's": "why is", 
				   "why've": "why have", "will've": "will have", "won't": "will not", 
				   "won't've": "will not have", "would've": "would have", "wouldn't": "would not", 
				   "wouldn't've": "would not have", "y'all": "you all", "y'all'd": "you all would",
				   "y'all'd've": "you all would have","y'all're": "you all are","y'all've": "you all have",
				   "you'd": "you would", "you'd've": "you would have", "you'll": "you will", 
				   "you'll've": "you will have", "you're": "you are", "you've": "you have" }
	expanded = ' '.join([contraction_mapping[t.lower()] if t.lower() in contraction_mapping else t for t in apostrophe_handled.split(" ")])
	return expanded

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

def remove_websites(text_tokens: List[str]) -> List[str]:
	result = []
	for i in range(len(text_tokens)):

		if text_tokens[i].lower().startswith('http'):
			text_tokens[i] = 'web'

	return result

def fix_spellings(text_tokens: List[str], spell: SpellChecker) -> List[str]:
	for i, w in enumerate(text_tokens):
		if w == ' ' or w == '':
			continue
		
		if w not in spell:
			c = spell.correction(w)

			if c == w:
				continue
			text_tokens[i] = c
			spellCheckerReplaced.append((w, c))

	return text_tokens

def initialize_spellchecker(language: str) -> SpellChecker:
	if language != 'en':
		return SpellChecker(language=language)

	spell = SpellChecker(language='en')

	# load word from additional dictionary
	from data.spellchecker.spellchecker import get_en_dictionary, get_organic_dictionary
	d = get_en_dictionary()
	spell.word_frequency.load_words(d)

	# load organic specific entities
	d = get_organic_dictionary()
	spell.word_frequency.load_words(d)

	return spell

spellCheckerReplaced = []

spell_checker_entities = [
	'Quora',
	'walmart',
	'costco',
	'kroger',
	'HEB',
	'Publix',
	'NPOP',
	'USDA',
	'cannot',
	'websites',
	'website',
	'die-offs',
	'googling',
	'zillions',
	'URL',
	'Facebook',
	'demonetized',
	'e. coli'
	"aren't",
	"ain't",
	'superbug',
	'superbugs'
	"you're",
	"don't",
	"doesn't",
	"Glyphosate",
	'maceration',
	'GMO',
	'non-GMO',
	'anti-GMO'
	'Cascadian',
	'Google',
	'24D',
	'odor',
	'16th',
	'RS10000',
	'Rs200',
	'reusing',
	'beets',
	'premade',
	'resilient',
	'gazillion',
	'TLDR',
	'to-do',
	'grass-fed',
	'pricy',
	'smellier',
	'FAQs',
	'omega-3',
	'non-vegetarian',
	'gmo-free',
	'conventionally-grown',
	'gluten-free',
	'manure-based',
	'organic-natural',
	'antibiotic-resistant',
	'organic-approved',
	'pesticide-free',
	'petroleum-based',
	'pro-organic',
	'all-natural',
	'environmentally-damaging',
	'food-grade',
	'gourmet-delight',
	'government-approved',
	'government-independent',
	'guilt-free',
	'locally-grown',
	'multi-cropping',
	'myth-busting',
	'natural-organic',
	'organically-grown',
	'organic-farming',
	'organic-inorganic',
	'peer-reviewed',
	'pesticides-herbicides',
	'safe-unsafe',
	'sludge-based',
	'three-fourths',
	'usdaac-credited',
	'youtube',
	'additive-free',
	'aflatoxin-free',
	'ago-chemical-dependent',
	'agro-complex',
	'agro-ecosystem',
	'amazonfresh',
	'animal-based',
	'animal-byproducts',
	'antibiotic-resistance',
	'antibiotics-fed',
	'antibiotics-hormones',
	'anti-nutrient',
	'antioxidant-rich',
	'apartments-like',
	'apple-strawberry-banana',
	'artificially-produced',
	'big-basket',
	'big-company',
	'big-picture',
	'bio-intensive',
	'blemish-free',
	'blood-meal',
	'brain building',
	'bucket-load',
	'budget-friendly',
	'bug-resistant',
	'carbon-footprint',
	'certified-noncertified',
	'chemical-free'
]