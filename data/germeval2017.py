import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import *
import torchtext
from torchtext import data
from torchtext.vocab import Vectors, GloVe, CharNGram
from torchtext.datasets import SequenceTaggingDataset, CoNLL2000Chunking
from prettytable import PrettyTable
from stop_words import get_stop_words

from data.custom_fields import ReversibleField
from data.custom_datasets import CustomGermEval2017Dataset
from data.data_loader import get_embedding


logger = logging.getLogger(__name__)

def preprocess_word(word: str) -> str:
    # TODO: Actual processing
    return word

def preprocess_relevance_word(word: str) -> int:
    if word == 'false':
        return 0
    return 1

def germeval2017_dataset(
                    pretrained_vectors,
                    batch_size=80,
                    root='./germeval2017',
                    train_file='train_v1.4.tsv',
                    validation_file='dev_v1.4.tsv',
                    test_file=None,
                    use_cuda=False,
                    use_stop_words=True):
    if use_stop_words:
        stop_words = get_stop_words('de')
    else:
        stop_words = []

    # contains the sentences                
    comment_field = ReversibleField(
                            batch_first=True,    # produce tensors with batch dimension first
                            lower=True,
                            sequential=True,
                            use_vocab=True,
                            init_token=None,
                            eos_token=None,
                            is_target=False,
                            stop_words=stop_words,
                            preprocessing=data.Pipeline(preprocess_word))

    relevant_field = data.Field(
                            batch_first=True,
                            is_target=True,
                            sequential=False,
                            use_vocab=False,
                            unk_token=None,
                            preprocessing=data.Pipeline(preprocess_relevance_word))

    general_sentiment_field = ReversibleField(
                            batch_first=True,
                            is_target=True,
                            sequential=False,
                            init_token=None,
                            eos_token=None,
                            unk_token=None,
                            use_vocab=True)

    padding_field = data.Field(
                            batch_first=True,
                            sequential=True,
                            use_vocab=True,
                            init_token=None,
                            eos_token=None,
                            unk_token=None,
                            is_target=False)

    fields = [
        (None, None),                                       # link to comment eg: (http://twitter.com/lolloseb/statuses/718382187792478208)
        ('comments', comment_field),                         # comment itself e.g. (@KuttnerSarah @DB_Bahn Hund = Fahrgast, Hund in Box = Gepäck.skurril, oder?)
        ('relevance', relevant_field),                      # is comment relevant true/false
        ('general_sentiments', general_sentiment_field),     # sentiment of comment (positive, negative, neutral)
        (None, None),                                        # apsect based sentiment e.g (Allgemein#Haupt:negative Sonstige_Unregelmässigkeiten#Haupt:negative Sonstige_Unregelmässigkeiten#Haupt:negative)
        ('padding', padding_field)                          # artificial field that we append to fill it with the padding information later to create the masks
            ]

    train, val, test = CustomGermEval2017Dataset.splits(
                            path=root,
                            train=train_file,
                            validation=validation_file,
                            test=test_file,
                            separator='\t',
                            fields=fields
    )

    # use updated fields
    fields = train.fields
    comment_field.build_vocab(train.comments, val.comments, test.comments, vectors=[pretrained_vectors])
    general_sentiment_field.build_vocab(train.general_sentiments)
    padding_field.build_vocab(train.padding, val.comments, test.comments)

    # build aspect fields
    for s_cat, f in train.aspect_sentiment_fields:
        f.build_vocab(train.__getattr__(s_cat), val.__getattr__(s_cat), test.__getattr__(s_cat))

    train_device = torch.device('cuda:0' if torch.cuda.is_available() and use_cuda else 'cpu')
    train_iter, val_iter, test_iter = data.BucketIterator.splits(
        (train, val, test), batch_size=batch_size, device=train_device)

    # add embeddings
    embedding_size = comment_field.vocab.vectors.shape[1]
    source_embedding = get_embedding(comment_field.vocab, embedding_size)

    examples = train.examples[0:3] + val.examples[0:3] + test.examples[0:3]

    return {
        'task': 'germeval2017',
        'split_length': (len(train), len(val), len(test)),
        'iters': (train_iter, val_iter, test_iter), 
        'vocabs': (comment_field.vocab, general_sentiment_field.vocab),
        'fields': [comment_field, relevant_field, general_sentiment_field, padding_field],
        'source_field_name': 'comments',
        'target_field_name': 'general_sentiments',
        'padding_field_name': 'padding',
        'examples': examples,
        'embeddings': (source_embedding, None),
        'dummy_input': Variable(torch.zeros((batch_size, 42), dtype=torch.long))
        }