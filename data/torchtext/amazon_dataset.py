import os
import string
from typing import Dict, List, Tuple, Union
import pickle
from collections import defaultdict
import re
import torchtext.data as data
import torch.utils.data
from misc.utils import create_dir_if_necessary, check_if_file_exists
from tqdm import tqdm

from data.torchtext.custom_fields import ReversibleField
from data.torchtext.custom_datasets import *
import pandas as pd

def add_tr_prefixes(path:str, tr_1=False, tr_2=False, tr_3=False, sp=False) -> str:
	fn = path.split('.')

	if sp:
		fn[0] += '_sp'

	if tr_1:
		fn[0] += '_TR-1'

	if tr_2:
		fn[0] += '_TR-2'

	if tr_3:
		fn[0] += '_TR-3' 
	return '.'.join(fn)

class AmazonDataset(Dataset):

	@staticmethod
	def sort_key(example):
		for attr in dir(example):
			if not callable(getattr(example, attr)) and \
					not attr.startswith("__"):
				return len(getattr(example, attr))
		return 0

	@classmethod
	def splits(cls, path=None, root='.data', train=None, validation=None,
			   test=None, hp=None, **kwargs) -> Tuple[Dataset]:
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

		# depending on the hps we have to adjust the paths
		train = add_tr_prefixes(train, hp.token_removal_1, hp.token_removal_2, hp.token_removal_3,  hp.use_spell_checkers)
		validation = add_tr_prefixes(validation, hp.token_removal_1, hp.token_removal_2, hp.token_removal_3, hp.use_spell_checkers)
		test = add_tr_prefixes(test, hp.token_removal_1, hp.token_removal_2, hp.use_spell_checkers)

		train_data = None if train is None else cls(
			os.path.join(path, train), hp=hp, **kwargs)
		# make sure, we use exactly the same fields across all splits
		train_aspects = train_data.aspects

		val_data = None if validation is None else cls(
			os.path.join(path, validation), a_sentiment=train_aspects, hp=hp, **kwargs)

		test_data = None if test is None else cls(
			os.path.join(path, test), a_sentiment=train_aspects, hp=hp, **kwargs)

		return tuple(d for d in (train_data, val_data, test_data)
					 if d is not None)

	def __init__(self, path, fields, a_sentiment=[], hp=None, **kwargs):
		self.aspect_sentiment_fields = []
		self.aspects = a_sentiment if len(a_sentiment) > 0 else []
		self.stats = defaultdict(get_stats_dd)
		self.na_labels = 0
		self.hp = None

		self.dataset_name = 'amazon'

		if hp.use_spell_checkers:
			self.dataset_name += '_SP'
		
		self.dataset_name += f'_{hp.clip_comments_to}'


		# first, try to load all models from cache
		filename = path.split("\\")[-1]
		examples, loaded_fields = self._try_load(filename.split(".")[0], fields)

		if not examples:
			examples, fields = self._load(path, filename, fields, a_sentiment, hp=hp, **kwargs)
			self._save(filename.split(".")[0], examples)
		else:
			fields = loaded_fields
			
		super(AmazonDataset, self).__init__(examples, tuple(fields))    

	def _load(self, path, filename, fields, a_sentiment=[], verbose=True, hp=None, **kwargs):
		examples = []
		self.hp = hp
		
		# remove punctuation
		punctuation_remover = str.maketrans('', '', string.punctuation + '…' + "“" + "„" + "‘")

		# In the end, those are the fields
		# The file has the aspect sentiment at the first aspect sentiment position
		# 1: Comment
		# 4: Apsect Specific sentiment List
		# 5: Padding Field
		
		aspect_sentiment_categories = set()
		aspect_sentiments: List[Dict[str, str]] = []

		raw_examples: List[List[Union[str, List[Dict[str, str]]]]] = []

		# read the pickeled dataframe
		df = pd.read_pickle(path)

		# iterate over all samples
		total_samples = df.count()['overall']
		iterator = tqdm(df.itertuples(), desc=f'Load {filename[0:7]}', leave=True, total=total_samples)


		for row in iterator:
			columns = []

			sentiment_dict = dict()
			aspect_category = getattr(row, 'aspect')
			aspect_sentiment = getattr(row, 'sentiment')
			comment = getattr(row, 'reviewText')			
			self.stats[aspect_category][aspect_sentiment] += 1					
			aspect_sentiment_categories.add(aspect_category)
			sentiment_dict[aspect_category] = aspect_sentiment


			# replace urls with regex
			comment = replace_urls_regex(comment)

			comment = comment.split(' ')

			# remove all empty entries
			comment = [w for w in comment if w.strip() != '']

			if hp.replace_url_tokens:
				comment = replace_urls(comment)
				
			comment = ' '.join(comment)

			columns = [
				comment,
				sentiment_dict,
				'' # padding
			]

			raw_examples.append(columns)

		# process the aspect sentiment
		if len(self.aspects) == 0:
			self.aspects = list(aspect_sentiment_categories)

			# make sure the list is sorted. Otherwise we'll have a different
			# order every time and can not transfer models
			self.aspects = sorted(self.aspects)

			# construct the fields
			fields = self._construct_fields(fields)

		for raw_example in tqdm(raw_examples, leave=True, desc='Constructing Aspects'):
			# go through each aspect sentiment and add it at the corresponding position
			ss = ['n/a'] * len(self.aspects)
			nas = len(self.aspects)
			for s_category, s in raw_example[1].items():
				pos = self.aspects.index(s_category)
				ss[pos] = s
				nas -= 1

			self.na_labels += nas
			raw_example[1] = ss
			
			# construct example and add it
			example = raw_example + ss
			examples.append(data.Example.fromlist(example, tuple(fields)))

		# clip comments
		for example in tqdm(examples, leave=True, desc='Clipping comments'):
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

		path = os.path.join(os.getcwd(), 'data', 'data', 'cache', self.dataset_name)
		create_dir_if_necessary(path)
		samples_path = os.path.join(path, name + ".pkl")
		aspects_path = os.path.join(path, name + "_aspects.pkl")

		# print(f'Trying to save loaded dataset to {samples_path}.')
		with open(samples_path, 'wb') as f:
			pickle.dump(samples, f)
			# print(f'Model {name} successfully saved.')

		with open(aspects_path, "wb") as f:
			pickle.dump(self.aspects, f)


	