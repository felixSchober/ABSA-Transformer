import numpy as np
import torch
from torchtext.data.field import Field
from torchtext.data.dataset import Dataset
import os

class ReversibleField(Field):
	def __init__(self, **kwargs):
		if kwargs.get('tokenize') is list:
			self.use_revtok = False
		else:
			self.use_revtok = True
		if kwargs.get('tokenize') is None:
			kwargs['tokenize'] = 'revtok'
		if 'unk_token' not in kwargs:
			kwargs['unk_token'] = ' UNK '
		super(ReversibleField, self).__init__(**kwargs)

	def reverse(self, batch, detokenize=True):
		if self.use_revtok:
			try:
				import revtok
			except ImportError:
				print("Please install revtok.")
				raise
		if not self.batch_first:
			batch = batch.t()
		with torch.cuda.device_of(batch):
			batch = batch.tolist()
		batch = [[self.vocab.itos[ind] for ind in ex] for ex in batch]  # denumericalize

		def trim(s, t):
			sentence = []
			for w in s:
				if w == t:
					break
				sentence.append(w)
			return sentence

		batch = [trim(ex, self.eos_token) for ex in batch]  # trim past frst eos

		if not detokenize:
			return batch

		def filter_special(tok):
			return tok not in (self.init_token, self.pad_token)

		batch = [filter(filter_special, ex) for ex in batch]
		if self.use_revtok:
			#test = revtok.detokenize(batch[0])
			return [revtok.detokenize(ex) for ex in batch]
		return [' '.join(ex) for ex in batch]

class ElmoField(Field):

	def __init__(self, language, hp, **kwargs):
		super(ElmoField, self).__init__(**kwargs)
		self.language = language
		self.hp = hp
		self.dtype = torch.float32
		self.embeddder = self._initialize_elmo()		
		self.cache = {}

	def _initialize_elmo(self):
		from elmoformanylangs import Embedder
		path = os.path.join(os.getcwd(), '.vector_cache', 'elmo', self.language + '.model')
		return Embedder(path, self.hp.batch_size)

	# def build_vocab(self, *args, **kwargs):
	# 	sources = []
	# 	for arg in args:
	# 		if isinstance(arg, Dataset):
	# 			sources += [getattr(arg, name) for name, field in
	# 						arg.fields.items() if field is self]
	# 		else:
	# 			sources.append(arg)
	# 	for data in sources:
	# 		if kwargs.get('verbose') is None or kwargs.get('verbose') == False:
	# 			iterator = data
	# 		else:
	# 			from tqdm.autonotebook import tqdm
	# 			data_list = [s for s in data]
	# 			iterator = tqdm(data_list)


	# 		for x in iterator:
	# 			if not self.sequential:
	# 				x = [x]
	# 			self._get_elmo([x])
				
	def _get_elmo(self, x):
		result = []
		for s in x:
			sentence = ' '.join(s)
			if sentence in self.cache:
				result.append(self.cache[sentence])
				continue
			arr = self.embeddder.sents2elmo([s])
			arr = arr[0]
			self.cache[sentence] = arr
			result.append(arr)
		return result

	def numericalize(self, arr, device=None):
		if self.include_lengths and not isinstance(arr, tuple):
			raise ValueError("Field has include_lengths set to True, but "
							 "input data is not a tuple of "
							 "(data batch, batch lengths).")
		if isinstance(arr, tuple):
			arr, lengths = arr
			lengths = torch.tensor(lengths, dtype=self.dtype, device=device)

		# use elmo to convert the sentences (arr) to an array
		arr = np.array(self.embeddder.sents2elmo(arr))
		arr = torch.as_tensor(arr, dtype=self.dtype, device=device)

		if self.sequential and not self.batch_first:
			arr.t_()
		if self.sequential:
			arr = arr.contiguous()

		if self.include_lengths:
			return arr, lengths
		return arr