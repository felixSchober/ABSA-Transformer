from torchtext import data, datasets, vocab
import torch
from prettytable import PrettyTable
from misc.run_configuration import RunConfiguration

from misc.utils import get_class_variable_table

# see https://github.com/mjc92/TorchTextTutorial/blob/master/01.%20Getting%20started.ipynb

def get_embedding(vocabulary, embedding_size):
    embedding = torch.nn.Embedding(len(vocabulary), embedding_size)
    embedding.weight.data.copy_(vocabulary.vectors)
    return embedding

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
        self.exaples = []

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

        self.trivial_accuracy = 0.0
    
    def load_data(self,
                loader,                
                custom_preprocessing: data.Pipeline=DEFAULT_DATA_PIPELINE):

        self.logger.info(f'Getting {self.pretrained_word_embeddings} with dimension {self.pretrained_word_embeddings_dim}')
        word_vectors: vocab
        if self.pretrained_word_embeddings == 'glove':
            word_vectors = vocab.GloVe(name=self.pretrained_word_embeddings_name, dim=self.pretrained_word_embeddings_dim)
        self.logger.info('Word vectors successfully loaded.')
                
        self.logger.debug('Start loading dataset')
        self.dataset = loader(
            word_vectors,
            self.batch_size,
            self.data_path,
            self.train_file,
            self.valid_file,
            self.test_file,
            self.use_cuda,
            self.use_stop_words)

        self.vocabs = self.dataset['vocabs']
        self.task = self.dataset['task']
        self.split_length = self.dataset['split_length']
        self.train_iter, self.valid_iter, self.test_iter = self.dataset['iters']
        self.fields = self.dataset['fields']
        self.exaples = self.dataset['examples']
        self.embedding = self.dataset['embeddings']
        self.dummy_input = self.dataset['dummy_input']
        self.source_field_name = self.dataset['source_field_name']
        self.target_field_name = self.dataset['target_field_name']
        self.padding_field_name = self.dataset['padding_field_name']

        self.target_size = len(self.vocabs[self.target_vocab_index])
        self.source_embedding = self.embedding[self.source_index]
        self.class_labels = list(self.vocabs[self.target_vocab_index].itos)

        self.source_reverser = self.fields[self.source_index]
        self.target_reverser = self.fields[self.target_vocab_index]

        self.log_parameters()
        self.show_stats()

        self.logger.info('Dataset loaded. Ready for training')

    def log_parameters(self):
        parameter_table = get_class_variable_table(self, 'Data Loader')
        self.logger.info(parameter_table)

    def show_stats(self):
        stats = self._show_split_stats()
        self.logger.info(stats)
        print(stats)

        stats = self._show_field_stats()
        self.logger.info(stats)
        print(stats)

        stats = self._calculate_dataset_stats(self.vocabs[1], 'General Sentiment')
        self.logger.info(stats)
        print(stats)

    def _show_split_stats(self) -> str:
        t = PrettyTable(['Split', 'Size'])
        t.add_row(['train', self.split_length[0]])
        t.add_row(['validation', self.split_length[1]])
        t.add_row(['test', self.split_length[2]])

        result = t.get_string(title='GERM EVAL 2017 DATASET')
        return result

    def _show_field_stats(self):
        t = PrettyTable(['Vocabulary', 'Size'])
        t.add_row(['Comments', len(self.fields[0].vocab)])
        t.add_row(['Relevant', 2])
        t.add_row(['General Sentiment', len(self.fields[2].vocab)])
        t.add_row(['Padding', 2])

        result = t.get_string(title='Vocabulary Stats')
        return result

    def _calculate_dataset_stats(self, target_vocab, title=None):
        total_samples = 0

        result_str = ''

        t = PrettyTable(['Label', 'Samples'])
        for l, freq in target_vocab.freqs.items():
            t.add_row([l, freq])
            total_samples += freq
        t.add_row(['Sum', total_samples])
        result_str = t.get_string(title=title)

        # trivial classifier
        t = PrettyTable(['Label', 'Triv. Accuracy'])

        for l, freq in target_vocab.freqs.items():
            acc = float(freq) / float(total_samples)
            self.trivial_accuracy = max(self.trivial_accuracy, acc)
            t.add_row([l, acc*100])

        result_str += '\n\n' + t.get_string(title=title)

        self.total_samples = total_samples
        return result_str

