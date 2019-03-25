import os
from typing import List
from misc.utils import check_if_file_exists
import pickle
import logging

logger = logging.getLogger(__name__)


def get_en_dictionary() -> List[str]:
	# check if file was already created
	path = os.path.join(os.getcwd(), 'data', 'spellchecker', 'hunspell-en_US-2018', 'en_US.pkl')
	if not check_if_file_exists(path):
		logger.info(f'Pre pickeled spellchecker dictionary does not exist at {path}.')
		# load source dict and process it
		src_dict_path = os.path.join(os.getcwd(), 'data', 'spellchecker', 'hunspell-en_US-2018', 'en_US.dic')
		if not check_if_file_exists(src_dict_path):
			logger.error('Could not find source spellchecker file at path {src_dict_path}. Please download it from the website.')
			raise ValueError(f'Source spellchecker file at path {src_dict_path} was not found. Please download it from the website.')

		# process file
		dictionary = []
		with open(src_dict_path, encoding='utf8') as input_file:
			first_line = True
			for line in input_file:
				if first_line:
					first_line = False
					continue
				parts = line.split('/')
				line = parts[0].replace('\n', '')
				dictionary.append(line)

		# save dict
		with open(path, 'wb') as f:
			pickle.dump(dictionary, f)
		logger.info(f'File successfully loaded and created. It is located at {path}')
		return dictionary

	# file exists, load it and return it
	with open(path, 'rb') as f:
		dictionary = pickle.load(f)

	if dictionary:
		logger.info(f'Dictionary successfully unpickeld. Loaded {len(dictionary)} words')
	return dictionary

def get_organic_dictionary() -> List[str]:
	# load organic specific entities
	path = os.path.join(os.getcwd(), 'data', 'spellchecker', 'organic-words.txt')

	if not check_if_file_exists(path):
		logger.error(f'Could not find source spellchecker file at path {path}. Please download it from the website.')
		return []
	dictionary = []
	with open(path, encoding='utf8') as input_file:
			for line in input_file:
				dictionary.append(line.replace('\n', ''))

	return dictionary
