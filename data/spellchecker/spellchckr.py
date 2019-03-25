import os
from typing import List
from misc.utils import check_if_file_exists
import pickle

def get_dictionary() -> List[str]:
	# check if file was already created
	path = os.path.join(os.getcwd(), 'data', 'spellchecker', 'hunspell-en_US-2018', 'en_US.pkl')
	if not check_if_file_exists(path):
		# load source dict and process it
		src_dict_path = os.path.join(os.getcwd(), 'data', 'spellchecker', 'hunspell-en_US-2018', 'en_US.dic')
		if not check_if_file_exists(src_dict_path):
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

		return dictionary

	# file exists, load it and return it
	with open(path, 'rb') as f:
		dictionary = pickle.load(f)
	return dictionary
