
import os
import string
from typing import Dict, List, Tuple, Union
import pickle
import re

import torchtext.data as data
from data.torchtext.custom_fields import ReversibleField
import torch.utils.data
from spellchecker import SpellChecker
from tqdm import tqdm
import spacy

from misc.utils import create_dir_if_necessary, check_if_file_exists
from data.torchtext.custom_datasets import *


ORGANIC_TASK_ALL = 'all'
ORGANIC_TASK_ENTITIES = 'entities'
ORGANIC_TASK_ATTRIBUTES = 'attributes'
ORGANIC_TASK_COARSE = 'coarse'

ORGANIC_TASK_ALL_COMBINE = 'all_combine'
ORGANIC_TASK_ENTITIES_COMBINE = 'entities_combine'
ORGANIC_TASK_ATTRIBUTES_COMBINE = 'attributes_combine'
ORGANIC_TASK_COARSE_COMBINE = 'coarse_combine'

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
	'gg': 'GMOs genetic engineering general',
	'': ''											# when there is no aspect (in case of irrelevant comments)
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
	'e': 'environment',
	'av': 'availability',
	'a': 'animal welfare',
	'pp': 'productivity',
	'': ''											# when there is no aspect (in case of irrelevant comments)
}

od_sentiment_mapping = {
	'0': 'neutral',
	'p': 'positive',
	'n': 'negative'
}

od_coarse_entities = {
	'g': 'organic',
	'p': 'organic',
	'f': 'organic',
	'c': 'organic',

	'cg': 'conventional',
	'cp': 'conventional',
	'cf': 'conventional',
	'cc': 'conventional',

	'gg': 'GMO'
}

od_coarse_attributes = {
	'g': 'general',
	'p': 'price',
	
	't': 'experienced quality',
	'q': 'experienced quality',

	's': 'safety and healthiness',
	'h': 'safety and healthiness',
	'c': 'safety and healthiness',

	'll': 'trustworthy sources',
	'or': 'trustworthy sources',
	'l': 'trustworthy sources',
	'av': 'trustworthy sources',

	'e': 'environment',
	'a': 'environment',
	'pp': 'environment',
}

def create_coarse_organic_mapping():
	result = {}
	for entity_key in od_entity_mapping.keys():
		if entity_key == '':
			continue
		for attribute_key in od_attribute_mapping.keys():
			if attribute_key == '':
				continue
			c_ent = od_coarse_entities[entity_key]
			c_att = od_coarse_attributes[attribute_key]
			compound_key = f'{entity_key}-{attribute_key}'
			result[compound_key] = f'{c_ent}: {c_att}'
	# add a 'missing aspect' key
	result[''] = ''
	return result

def get_all_mapping():
	result = {}
	for entity_key, entity in od_entity_mapping.items():
		if entity_key == '':
			continue
		for attribute_key, attribute in od_attribute_mapping.items():
			if attribute_key == '':
				continue
			compound_key = f'{entity_key}-{attribute_key}'
			result[compound_key] = f'{entity}: {attribute}'

	# add a 'missing aspect' key
	result[''] = ''
	return result



class OrganicDataset(Dataset):

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
			path=os.path.join(path, train), length=lengths[0], **kwargs)
		# make sure, we use exactly the same fields across all splits
		train_aspects = train_data.aspects

		val_data = None if validation is None else cls(
			path=os.path.join(path, validation), a_sentiment=train_aspects, length=lengths[1], **kwargs)

		test_data = None if test is None else cls(
			path=os.path.join(path, test), a_sentiment=train_aspects, length=lengths[2], **kwargs)

		return tuple(d for d in (train_data, val_data, test_data)
					 if d is not None)
	
	def __init__(self, name, path, fields, a_sentiment=[], separator='\t', task=None, hp=None, **kwargs):
		self.aspect_sentiment_fields = []
		self.aspects = a_sentiment if len(a_sentiment) > 0 else []
		self.dataset_name = name

		# add spellChecked if spell checker is active
		if hp.use_spell_checkers:
			self.dataset_name += '_SP'

		if hp.use_text_cleaner:
			self.dataset_name += '_TC'

		if hp.contraction_removal:
			self.dataset_name += '_CR'

		if hp.organic_text_cleaning:
			self.dataset_name += '_OC'

		self.dataset_name += f'_{hp.clip_comments_to}'

		# first, try to load all models from cache
		_, filename = os.path.split(path)
		filename = f'{filename.split(".")[0]}_{task}'

		examples, loaded_fields = self._try_load(filename, fields)

		if not examples:
			examples, fields = self._load(path, filename, fields, a_sentiment, separator, task=task, hp=hp, **kwargs)
			self._save(filename, examples)
		else:
			fields = loaded_fields
			
		super(OrganicDataset, self).__init__(examples, tuple(fields))    

	def _load(self, path, filename, fields, a_sentiment=[], separator='|', verbose=True, hp=None, task=None, length=None, **kwargs):
		examples = []
		
		

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

		# load organic spell checker
		if hp.organic_text_cleaning:
			from data.spellchecker.spellchecker import get_organic_words_replacement
			organic_text_cleaning_dict = get_organic_words_replacement()

		if hp.use_spell_checkers:
			spell = initialize_spellchecker(hp.language)
		else:
			spell = None

		if task.startswith(ORGANIC_TASK_ALL):
			aspect_example_index = -1
			mapping = get_all_mapping()
		elif task.startswith(ORGANIC_TASK_ENTITIES):
			aspect_example_index = -5
			mapping = od_entity_mapping
		elif task.startswith(ORGANIC_TASK_ATTRIBUTES):
			mapping = od_attribute_mapping
			aspect_example_index = - 4
		elif task.startswith(ORGANIC_TASK_COARSE):
			aspect_example_index = -1
			mapping = create_coarse_organic_mapping()

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

			comments = {}
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
					pass
					# continue

				# aspect sentiment is missing
				if len(columns) == 12:
					columns.append('')
					columns.append(dict())
					last_sample = columns
				else:
					# based on aspect task select columns
					aspect_category = columns[aspect_example_index].strip()
					aspect_category = aspect_category.replace(';', '').replace('"', '')

					# use mapping to get a more human readable name
					aspect_category = mapping[aspect_category]
					
					s_k = columns[7].strip()
					if s_k != '':
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
						if aspect_category != '':
							comment_sentiment_dict[aspect_category] = aspect_sentiment
						continue	

					if aspect_category != '':
						comment_sentiment_dict[aspect_category] = aspect_sentiment
					
				# remove punctuation and clean text
				last_sample = self.process_comment_text(last_sample, hp, organic_text_cleaning_dict, spell)
				
				# add aspect sentiment field
				last_sample.append('')

				# add padding field
				last_sample.append('')

		# convert the comment dictionary to raw example list
		raw_examples = self.convert_to_raw_examples(comments, hp)

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

			#example.comments = [w.strip() for w in example.comments]

			comment_length: int = len(example.comments)
			if comment_length > hp.clip_comments_to:
				example.comments = example.comments[:hp.clip_comments_to]
				comment_length = len(example.comments)

			example.padding = ['0'] * comment_length
		return examples, fields

	def process_comment_text(self, sample, hp, organic_text_cleaning_dict, spell):
		# remove punctuation and clean text
		comment = sample[-3]
		comment = comment.translate(punctuation_remover)

		# remove non ascii characters with empty space
		comment = re.sub(r'[^\x00-\x7f]',r' ', comment)
		
		if hp.contraction_removal:
			comment = en_contraction_removal(comment)

		comment = comment.split(' ')

		if hp.replace_url_tokens:
			comment = replace_urls(comment)

		if hp.organic_text_cleaning:
			comment = fix_organic_spelling(comment, organic_text_cleaning_dict)

		if hp.use_spell_checkers:
			comment = fix_spellings(comment, spell)

		comment = ' '.join(comment)
		if hp.use_text_cleaner:
			comment = text_cleaner(comment, hp.language, spell)

		sample[-3] = comment
		return sample

	def convert_to_raw_examples(self, comments, hp):
		raise NotImplementedError()
		
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


class SingleSentenceOrganicDataset(OrganicDataset):

	def __init__(self, **kwargs):
		super(SingleSentenceOrganicDataset, self).__init__('organic2019Sentence', **kwargs)


	def convert_to_raw_examples(self, comments, hp):
		raw_examples = []

		for comment_sentences in comments.values():
			for sentence in comment_sentences:
				raw_examples.append(sentence)
		return raw_examples


class DoubleSentenceOrganicDataset(OrganicDataset):

	def __init__(self, **kwargs):
		super(DoubleSentenceOrganicDataset, self).__init__('organic2019DoubleSentence', **kwargs)


	def convert_to_raw_examples(self, comments, hp):
		raw_examples = []
		max_dual_sentence_length = (hp.clip_comments_to // 2) - 1 # -1 because space between sentences
		for comment_sentences in comments.values():

			# 1st sentence per comment does not have a previous sentence
			s = comment_sentences[0]
			raw_examples.append(s)
			sentences = [s[-6] for s in comment_sentences]
			for i in range(len(comment_sentences) - 1):
				first_comment_text = sentences[i]
				second_comment_text = sentences[i+1]

				# clip both comments. 
				# clip the first comment at the front (since the last words are nearer at the current sentence)
				# clip the second comment at the back.
				# also, do not clip a word in half
				first_comment_text, second_comment_text = intelligent_sentences_clipping(first_comment_text, second_comment_text, max_dual_sentence_length)

				# prepend this text to the next comment and clip both comments
				comment_sentences[i+1][-6] = f'{first_comment_text} {second_comment_text}'
				raw_examples.append(comment_sentences[i+1])

		return raw_examples
		

def fix_organic_spelling(text_tokens: List[str], organic_text_cleaning_dict) -> List[str]:
	for i, w in enumerate(text_tokens):
		if w == ' ' or w == '':
			continue

		if w in organic_text_cleaning_dict:
			text_tokens[i] = organic_text_cleaning_dict[w]

	return text_tokens