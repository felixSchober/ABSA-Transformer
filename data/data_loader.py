import os
from typing import List
from torchtext import data, datasets, vocab
from torch.nn import Embedding
from prettytable import PrettyTable
from misc.run_configuration import RunConfiguration
import matplotlib as mpl
mpl.use('agg')
import matplotlib.pyplot as plt
import seaborn as sns
from misc.utils import get_class_variable_table, create_dir_if_necessary, check_if_file_exists
import pandas as pd



# see https://github.com/mjc92/TorchTextTutorial/blob/master/01.%20Getting%20started.ipynb

def get_embedding(vocabulary, embedding_size, embd):
	if embd == 'elmo':
		return None
	embedding = Embedding(len(vocabulary), embedding_size)
	embedding.weight.data.copy_(vocabulary.vectors)
	return embedding

def get_embedding_size(field, embd) -> int:
	if embd == 'elmo':
		return 1024
	else:
		return field.vocab.vectors.shape[1]

DEFAULT_DATA_PIPELINE = data.Pipeline(lambda w: '0' if w.isdigit() else w )

class Dataset(object):

	def __init__(self,
				name: str,
				logger,                
				configuration: RunConfiguration,
				source_index: int,
				target_vocab_index: int,
				data_path: str,
				train_file:str,
				valid_file:str,
				test_file:str,
				file_format: str,
				init_token: str = None,
				eos_token: str = None,
				):
		self.embedding = None
		self.train_iter = None
		self.valid_iter = None
		self.test_iter = None
		self.word_field = None
		self.name = name
		self.dataset = None
		self.vocabs = []
		self.task = ''
		self.examples = []

		self.configuration = configuration
		self.batch_size = configuration.batch_size
		self.language = configuration.language
		self.data_path = data_path
		self.train_file = train_file
		self.valid_file = valid_file
		self.test_file = test_file
		self.file_format = file_format
		self.init_token = init_token
		self.eos_token = eos_token
		self.pretrained_word_embeddings = configuration.embedding_type
		self.pretrained_word_embeddings_dim = configuration.embedding_dim
		self.pretrained_word_embeddings_name = configuration.embedding_name
		self.use_cuda = configuration.use_cuda
		self.use_stop_words = configuration.use_stop_words
		self.clip_comments_to = configuration.clip_comments_to
		self.logger = logger
		self.split_length = (0, 0, 0)
		self.total_samples = -1

		self.source_index = source_index
		self.target_vocab_index = target_vocab_index
		self.target_size = -1
		self.source_embedding = None
		self.class_labels = None
		self.source_field_name: str = ''
		self.target_field_name: str = ''
		self.padding_field_name: str = ''
		self.source_reverser = None
		self.target_reverser = None
		self.baselines = {}
		self.verbose = True

		# for each transformer head this list contains a list of class weight values
		self.class_weights: List[List[float]] = []

		# this list contains a loss scaling for each transformer head. This uses the inverse weight of the 
		# number of times a label counts as n/a
		self.t_heads_weights: List[float] = []

		self.majority_class_baseline = 0.0
		

	def load_data(self,
				loader,                
				custom_preprocessing: data.Pipeline=DEFAULT_DATA_PIPELINE,
				verbose=True):

		self.verbose = verbose

		if self.verbose:
			# create an image folder
			self.img_stats_folder = os.path.join(self.data_path, 'stats')
			create_dir_if_necessary(self.img_stats_folder)

		self.logger.info(f'Getting {self.pretrained_word_embeddings} with dimension {self.pretrained_word_embeddings_dim}')
		word_vectors: vocab
		word_vectors = None
		if self.pretrained_word_embeddings == 'glove':
			word_vectors = vocab.GloVe(name=self.pretrained_word_embeddings_name, dim=self.pretrained_word_embeddings_dim)
		elif self.pretrained_word_embeddings == 'fasttext':
			word_vectors = vocab.FastText(language=self.language)
		self.logger.info('Word vectors successfully loaded.')
				
		self.logger.debug('Start loading dataset')
		self.dataset = loader(
			self.name,
			word_vectors,
			self.configuration,
			self.batch_size,
			self.data_path,
			self.train_file,
			self.valid_file,
			self.test_file,
			self.use_cuda,
			self.verbose)

		self.vocabs = self.dataset['vocabs']
		self.task = self.dataset['task']
		self.ds_stats = self.dataset['stats']
		self.split_length = self.dataset['split_length']
		self.train_iter, self.valid_iter, self.test_iter = self.dataset['iters']
		self.fields = self.dataset['fields']
		self.target = self.dataset['target']
		self.target_names = [n for n, _ in self.target]
		self.examples = self.dataset['examples']
		self.embedding = self.dataset['embeddings']
		self.dummy_input = self.dataset['dummy_input']
		self.source_field_name = self.dataset['source_field_name']
		self.target_field_name = self.dataset['target_field_name']
		self.padding_field_name = self.dataset['padding_field_name']
		self.baselines = self.dataset['baselines']

		self.target_size = len(self.vocabs[self.target_vocab_index])
		self.source_embedding = self.embedding[self.source_index]
		self.class_labels = list(self.vocabs[self.target_vocab_index].itos)

		self.source_reverser = self.dataset['source_field']
		self.target_reverser = self.target[0]
		self.log_parameters()

		if verbose:
			# sns.set(style="whitegrid")
			sns.set_style("white")
			sns.despine()

			sns.set_color_codes()
			# sns.set_context("paper")
			sns.set(rc={"font.size":18,"axes.labelsize":22})
			# sns.set(font_scale=1.7)
			self.show_stats()
		else:
			self._calculate_dataset_stats()

		self.logger.info('Dataset loaded. Ready for training')

	def log_parameters(self):
		parameter_table = get_class_variable_table(self, 'Data Loader')
		self.logger.info('\n' + parameter_table)

	def show_stats(self):
		try:
			stats = self._show_split_stats()
			self.logger.info('\n' + stats)
			print(stats)

			stats = self._show_field_stats()
			self.logger.info('\n' + stats)
			print(stats)
		except Exception as err:
			self.logger.exception('Could not show dataset stats')
		
		stats = self._calculate_dataset_stats()

		try:
			self.logger.info('\n' + stats)
			print(stats)
			stats = self._show_ds_split_stats()
			self.logger.info('\n' + stats)
			print(stats)
		except Exception as err:
			self.logger.exception('Could not print dataset stats')

	def _show_split_stats(self) -> str:
		t = PrettyTable(['Split', 'Size'])
		t.add_row(['train', self.split_length[0]])
		t.add_row(['validation', self.split_length[1]])
		t.add_row(['test', self.split_length[2]])

		result = t.get_string(title='GERM EVAL 2017 DATASET')
		return result

	def _show_ds_split_stats(self):

		if self.ds_stats is None:
			return ''

		result = ''
		split_names = ('Train', 'Validation', 'Test')

		for iter_stats, split_name in zip(self.ds_stats, split_names):
			pos = 0
			neg = 0
			neu = 0
			t = PrettyTable(['Category', 'POS', 'NEG', 'NEU', 'Sum'])
			for cat_name, sentiments in iter_stats.items():
				pos += sentiments['positive']
				neg += sentiments['negative']
				neu += sentiments['neutral']
				t.add_row([cat_name, sentiments['positive'], sentiments['negative'], sentiments['neutral'], sum(sentiments.values())])
			t.add_row(['Total', pos, neg, neu, (pos + neg + neu)])
			result += t.get_string(title=split_name) + '\n\n'

		return result

	def _show_field_stats(self):
		t = PrettyTable(['Vocabulary', 'Size'])
		for (name, f) in self.fields.items():
			if name is None or not f.use_vocab:
				continue
			t.add_row([name, len(f.vocab)])

		result = t.get_string(title='Vocabulary Stats')
		return result

	def _calculate_dataset_stats(self):
		self.class_weights = []
		result_str = '\n\n'
		target_sentiment_distribution = []
		target_sentiment_samples = []
		target_sentiment_distribution_labels = []
		for name, f in self.target:
			if name is None or not f.use_vocab:
				continue
			total_samples = 0
			target_sentiment_distribution_labels.append(name)

			f_vocab = f.vocab

			not_na_samples = 0 # samples that are not n/a
			for l, freq in f_vocab.freqs.items():
				total_samples += freq

				if not l == 'n/a':
					not_na_samples += freq
			target_sentiment_samples.append(not_na_samples)

			majority_class_baseline = 0.0
			class_weight = [0.0] * self.target_size
			t = PrettyTable(['Label', 'Samples', 'Triv. Accuracy', 'Class Weight'])
			labels = []

			sentiment_distributions = []

			num_na_samples = total_samples - not_na_samples
			observation_distribution = [float(num_na_samples)/total_samples, float(not_na_samples)/total_samples]
			observation_distribution_lables = ['N/A', 'Sentiment']
			target_sentiment_distribution.append(observation_distribution[1])

			for l, freq in f_vocab.freqs.items():
				acc = float(freq) / float(total_samples)
				majority_class_baseline = max(majority_class_baseline, acc)
				stoi_pos = f_vocab.stoi[l]
				class_weight[stoi_pos] = 1 - acc

				if l != 'n/a':
					sentiment_distributions.append(float(freq) / not_na_samples)
					labels.append(l)

				t.add_row([l, freq, acc*100, class_weight[stoi_pos]])
			self.class_weights.append(class_weight)

			head_weight = 1.0 - not_na_samples / total_samples
			self.t_heads_weights.append(head_weight)

			t.add_row(['Sum', total_samples, '', 1.0])
			t.add_row(['Head Weight', '', '', head_weight])

			if self.verbose:
				self.plot_dataset_stats(sentiment_distributions, labels, f'Sentiment Distribution - {name}', f'{name} sentiments.pdf')
				self.plot_dataset_stats(observation_distribution, observation_distribution_lables, f'Ratio of N/A and sentiment - {name}', f'{name} observations.pdf')

			if not 'majority_class' in self.baselines:
				self.baselines['majority_class'] = majority_class_baseline
				self.majority_class_baseline = majority_class_baseline
				self.total_samples = total_samples

			result_str += '\n\n' + t.get_string(title=name) + '\n\n'

		if self.verbose:
			self.plot_dataset_stats(target_sentiment_samples, target_sentiment_distribution_labels, f'Dataset Aspects - Distribution', 'aspect_distribution.pdf')

		return result_str

	def plot_dataset_stats(self, samples, labels, title, fileName):
		path = os.path.join(self.img_stats_folder, fileName)
		# don't generate if already exists
		if check_if_file_exists(path):
			return

		try:
			df = pd.DataFrame({
				'Samples': samples,
				'Aspect': labels
			})

			plt.figure(figsize=(20,10))
			ax = sns.barplot(data=df, color='b', x='Aspect', y='Samples')
			plt.title(title, fontsize=20) 
			plt.xticks(rotation=45, ha="right")
			ax.get_yaxis().get_major_formatter().set_scientific(False)
			plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
			ax.yaxis.set_major_formatter(mpl.ticker.StrMethodFormatter('{x:,.0f}'))

			plt.savefig(path, format=fileName.split('.')[-1])
		except Exception as err:
			self.logger.exception('Could not plot ' + title)


		# IF YOU WANT TO HAVE A PIE, USE THIS
		# vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv
		# fig1, ax1 = plt.subplots()
		# patches, texts, autotexts = ax1.pie(fractions, pctdistance=0.85, labels=labels, autopct='%1.1f%%',
		# 		shadow=False, startangle=90)

		# for text in texts:
		# 	text.set_color('grey')
		# for autotext in autotexts:
		# 	autotext.set_color('white')

		# centre_circle = plt.Circle((0,0),0.70,fc='white')
		# fig = plt.gcf()
		# fig.gca().add_artist(centre_circle)
		# ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

		# plt.title(title)
		# plt.tight_layout(pad=0.8, w_pad=0.8, h_pad=1.0)
		# plt.savefig(path, format=fileName.split('.')[-1])
