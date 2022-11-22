import torchtext.data as data
import logging
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
from collections import defaultdict

import torch.utils.data

from torchtext.data.utils import RandomShuffler
from torchtext.utils import download_from_url, unicode_csv_reader
import spacy
from spellchecker import SpellChecker

from misc.utils import check_if_file_exists

logger = logging.getLogger(__name__)

# remove punctuation
punctuation_remover = str.maketrans('', '', string.punctuation + '…' + "“" + "„")
url_regex = r'(?:http(s)?:\/\/)?[\w.-]+(?:\.[\w\.-]+)+[\w\-\._~:/?#[\]@!\$&\(\)\*\+,;=.]+'

def get_stats_dd():
			return {'positive': 0, 'neutral': 0, 'negative': 0}

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
		if 'fields' not in kwargs.keys():
			kwargs['fields'] = train_data.fields

		val_data = None if validation is None else cls(
			os.path.join(path, validation), **kwargs)

		test_data = None if test is None else cls(
			os.path.join(path, test), **kwargs)

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
	def is_within_directory(directory, target):
		
		abs_directory = os.path.abspath(directory)
		abs_target = os.path.abspath(target)
	
		prefix = os.path.commonprefix([abs_directory, abs_target])
		
		return prefix == abs_directory
	
	def safe_extract(tar, path=".", members=None, *, numeric_owner=False):
	
		for member in tar.getmembers():
			member_path = os.path.join(path, member.name)
			if not is_within_directory(path, member_path):
				raise Exception("Attempted Path Traversal in Tar File")
	
		tar.extractall(path, members, numeric_owner=numeric_owner) 
		
	
	safe_extract(tar, path=path, members=dirs)
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

	
	def fix_spellings(self, text_tokens: List[str], spell: SpellChecker, language='en') -> List[str]:
		for i, w in enumerate(text_tokens):
			if w == ' ' or w == '':
				continue

			# don't replace if all caps
			if w.isupper():
				continue

			# check if it was already replaced
			if w in self.spellCheckerReplaced:
				text_tokens[i] = self.spellCheckerReplaced[w]
				continue
			
			if w not in spell:
				c = spell.correction(w)

				if c == w:
					continue
				text_tokens[i] = c
				self.spellCheckerReplaced[w] = c

		
		self.save_spellchecker_cache(language)
		return text_tokens

	def save_spellchecker_cache(self, language):
		path = os.path.join(os.getcwd(), 'data', 'spellchecker', language + '_cache.pkl')
		with open(path, "wb") as f:
				pickle.dump(self.spellCheckerReplaced, f)

	def load_spellchecker_cache(self, language):
		path = os.path.join(os.getcwd(), 'data', 'spellchecker', language + '_cache.pkl')
		if check_if_file_exists(path):
			with open(path, 'rb') as f:
				loaded = pickle.load(f)
				self.spellCheckerReplaced = loaded


	def initialize_spellchecker(self, language: str) -> SpellChecker:

		#try to initialize cache
		self.load_spellchecker_cache(language)

		if language != 'en':
			if language == 'de':
				spell = SpellChecker(language=None)

				from data.spellchecker.spellchecker import get_de_dictionary
				spell.word_frequency.load_words(germeval_words)
				spell.word_frequency.load_words(get_de_dictionary())
			else:
				spell = SpellChecker(language=language)

			return spell

		spell = SpellChecker(language='en')

		# load word from additional dictionary
		from data.spellchecker.spellchecker import get_en_dictionary, get_organic_dictionary
		d = get_en_dictionary()
		spell.word_frequency.load_words(d)

		# load organic specific entities
		d = get_organic_dictionary()
		spell.word_frequency.load_words(d)

		return spell


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

def text_cleaner(text: str, language: str):
	spacy_nlp = spacy.load(language)
	parsed = spacy_nlp(text)
	final_tokens = []
	for t in parsed:

		if t.is_punct or t.is_space or t.like_num or t.like_url or str(t).startswith('@'):
			continue
		
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

def replace_urls_regex(sentence: str, url_token: str = '<URL>') -> str:
	return re.sub(url_regex, url_token, sentence)

def replace_urls(words: List[str], url_token: str = '<URL>') -> List[str]:
	return [url_token if (w.lower().startswith('www') or w.lower().startswith('http')) else w for w in words]


def intelligent_sentences_clipping(s1: str, s2: str, clip_to: int):
	# first clip s1 at the front.
	# let's add words from the back until we hit the clipping mark

	clipped = []
	current_word_count = 0
	for w in reversed(s1.split(' ')):
		# what is the len of the current word
		
		if current_word_count < clip_to:
			current_word_count += 1

			# enter at front because we reversed the sentence
			clipped.insert(0, w)
		else:
			# doesn't fit anymore -> we are finished with the sentence
			break

	s1 = ' '.join(clipped)

	# try to fill with s2
	clip_to = (clip_to - len(clipped)) + clip_to
	
	# s2
	clipped = []
	current_word_count = 0
	for w in s2.split(' '):
		
		if current_word_count < clip_to:
			current_word_count += 1

			# enter at back because sentence is not reversed
			clipped.append(w)
		else:
			# doesn't fit anymore -> we are finished with the sentence
			break

	# now.. there is a weird case, where clip_comments is to short, so that not even the first words fit.
	# in this case, show a warning and clip the words
	if len(clipped) == 0:
		logger.warn('Clip comments to might be to low. Could not fit even one word into sentence. Sentence: ' + s2)
		s2 = s2[:clip_to]
	else:
		s2 = ' '.join(clipped)

	return s1, s2

	
def intelligent_sentence_clipping(s: str, clip_to: int) -> str:
	clipped = []
	current_char_count = 0
	for w in s:
		wl = len(w)
		spaces = len(clipped)
		if current_char_count + spaces + wl <= clip_to:
			current_char_count += wl

			# enter at back because sentence is not reversed
			clipped.append(w)
		else:
			# doesn't fit anymore -> we are finished with the sentence
			break

	if len(clipped) == 0:
		return s[:clip_to]

	return ' '.join(clipped)

germeval_words = [
	'db',
	'KVB',
	'ITB',
	'SSB',
	'MVG',
	'MVV'
]
