from data.data_loader import *
from data.transferLearning import transfer_learning

class TransferDataLoader(Dataset):

	def __init__(self,
				**args
				):
		super(TransferDataLoader, self).__init__(**args)

	def load_data(self,
				loader,                
				verbose=True):

		self.verbose = verbose

		self.logger.info(f'Getting {self.pretrained_word_embeddings} with dimension {self.pretrained_word_embeddings_dim}')
		word_vectors: vocab
		word_vectors = None
		if self.pretrained_word_embeddings == 'glove':
			word_vectors = vocab.GloVe(name=self.pretrained_word_embeddings_name, dim=self.pretrained_word_embeddings_dim)
		elif self.pretrained_word_embeddings == 'fasttext':
			word_vectors = vocab.FastText(language=self.language)
		self.logger.info('Word vectors successfully loaded.')
				
		self.logger.debug('Start loading dataset')
		self.dataset = transfer_learning(
			self.name,
			word_vectors,
			self.configuration,
			self.batch_size,
			loader,
			self.data_path,
			self.train_file,
			self.valid_file,
			self.test_file,
			self.use_cuda,
			self.verbose)

		num_datasets = len(loader)
		for i in range(num_datasets):

			self.vocabs = self.dataset['vocabs'][i]
			self.task = self.dataset['task']
			self.ds_stats = self.dataset['stats'][i]
			self.split_length = self.dataset['split_length'][i]
			self.train_iter, self.valid_iter, self.test_iter = self.dataset['iters'][i]
			self.fields = self.dataset['fields']
			self.target = self.dataset['target'][i]
			self.target_names = [n for n, _ in self.target[i]]
			self.examples = self.dataset['examples'][i]
			self.embedding = self.dataset['embeddings']
			self.dummy_input = self.dataset['dummy_input']
			self.source_field_name = self.dataset['source_field_name']
			self.target_field_name = self.dataset['target_field_name']
			self.padding_field_name = self.dataset['padding_field_name']
			self.baselines = self.dataset['baselines']

			self.target_size = len(self.vocabs[self.target_vocab_index][i])
			self.source_embedding = self.embedding[self.source_index]
			self.class_labels = list(self.vocabs[self.target_vocab_index][i].itos)

			self.source_reverser = self.dataset['source_field'][i]
			self.target_reverser = self.target[0]
			self.log_parameters()

			if verbose:
				self.show_stats()
			else:
				self._calculate_dataset_stats()

			self.logger.info('Dataset loaded. Ready for training')
			yield i
