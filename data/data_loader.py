from torchtext import data, datasets
import torch
import spacy

# see https://github.com/mjc92/TorchTextTutorial/blob/master/01.%20Getting%20started.ipynb

def tokenize(language: str):
    s = spacy.load(language)
    def tokenizer(text) -> list:
        token_text_list = [tok.text for tok in s.tokenizer(text)]
        return token_text_list
    return tokenizer

def get_embedding(vocabulary, embedding_size):
    embedding = torch.nn.Embedding(len(vocabulary), embedding_size)
    embedding.weight.data.copy_(vocabulary.vectors)
    return embedding

DEFAULT_DATA_PIPELINE = data.Pipeline(lambda w: '0' if w.isdigit() else w )

class DataLoader(object):

    def __init__(self, name):
        self.embedding = None
        self.train_iter = None
        self.valid_iter = None
        self.test_iter = None
        self.name = name
    
    def load_data(self,
                language: str,
                data_path: str,
                train_file:str,
                valid_file:str,
                test_file:str,
                file_format: str,
                init_token: str = '<SOS>',
                eos_token: str = '<EOS>',
                pretrained_word_embeddings: str='glove.6B.100d',
                use_cuda: bool=False,
                custom_preprocessing: data.Pipeline=DEFAULT_DATA_PIPELINE
    ):

        # language string like 'en' or 'de'

        # TODO: Create Doc String


        # get tokenizer function which accepts a text and generates tokens
        tokenizer = tokenize(language)

        # create field objects for preprocessing / tokenizing 
        # fields are classes which contain information on how the data
        # should be preprocessed. In addition, they also create the vocabulary
        TEXT = data.Field(sequential=True,
                            tokenize=tokenizer,
                            init_token=init_token, 
                            eos_token=eos_token,
                            batch_first=True,
                            lower=True)

        LABEL = data.Field(sequential=False, use_vocab=False)

        if train_file is None or valid_file is None or test_file is None:
            train, val, test = datasets.SST.splits(text_field=TEXT, label_field=LABEL)
        else:
            train, val, test = data.TabularDataset.splits(
            path=data_path, train=train_file,
            validation=valid_file, test=test_file, format=file_format,
            fields=[('Text', TEXT), ('Label', LABEL)])


        TEXT.build_vocab(train, vectors=pretrained_word_embeddings)

        device = 0 if use_cuda else -1

        self.train_iter, self.val_iter, self.test_iter = data.Iterator.splits(
            (train, val, test), sort_key=lambda x: len(x.Text),
            batch_sizes=(32, 256, 256), device=device)

        # TODO: Make variable
        self.embedding = get_embedding(TEXT.vocab, 100)

    
