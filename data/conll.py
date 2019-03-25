from typing import Tuple, List, Dict, Optional, Union, Iterable
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import *

import torchtext
from torchtext import data
from torchtext.datasets import SequenceTaggingDataset, CoNLL2000Chunking
from torchtext.vocab import Vectors, GloVe, CharNGram
import numpy as np
import random
import logging
import spacy
from data.data_loader import get_embedding
from data.torchtext.custom_fields import ReversibleField
from data.torchtext.custom_datasets import CustomSequenceTaggingDataSet

logger = logging.getLogger(__name__)

ExamplePair = Tuple[str, str] # x, y
ExampleSentence = List[ExamplePair]
ExampleList = List[ExampleSentence]

def preprocess_test(word):
    return '0' if word.isdigit() else word

def conll2003_dataset(tag_type, batch_size,
                          root='./conll2003', 
                          train_file='eng.train.txt', 
                          validation_file='eng.testa.txt',
                          test_file='eng.testb.txt',
                          convert_digits=True,
                          use_cuda=False):
    """
    https://github.com/kolloldas/torchnlp/blob/master/torchnlp/data/conll.py
    conll2003: Conll 2003 (Parser only. You must place the files)
    Extract Conll2003 dataset using torchtext. Applies GloVe 6B.200d and Char N-gram
    pretrained vectors. Also sets up per word character Field
    Parameters:
        tag_type: Type of tag to pick as task [pos, chunk, ner]
        batch_size: Batch size to return from iterator
        root: Dataset root directory
        train_file: Train filename
        validation_file: Validation filename
        test_file: Test filename
        convert_digits: If True will convert numbers to single 0's
    Returns:
        A dict containing:
            task: 'conll2003.' + tag_type
            iters: (train iter, validation iter, test iter)
            vocabs: (Inputs word vocabulary, Inputs character vocabulary, 
                    Tag vocabulary )
    """
    
    # Setup fields with batch dimension first
    inputs_word = ReversibleField(batch_first=True, lower=True,
                                preprocessing=data.Pipeline(preprocess_test))           

    # the label constits of three parts:
    #   - Part of speech tag
    #   - syntactic chunk tag   (I-TYPE)
    #   - named entity tag      (I-TYPE)
    labels = ReversibleField(batch_first=True, is_target=True)

    words_field = [('inputs_word', inputs_word)]
    labels_field = [('labels', labels) if label == tag_type else (None, None) 
                for label in ['pos', 'chunk', 'ner']]
    fields = ( words_field + labels_field)

    # Load the data
    train, val, test = CustomSequenceTaggingDataSet.splits(
                                path=root, 
                                train=train_file, 
                                validation=validation_file, 
                                test=test_file,
                                separator=' ',
                                fields=tuple(fields))

    logger.info('---------- CONLL 2003 %s ---------'%tag_type)
    logger.info('Train size: %d'%(len(train)))
    logger.info('Validation size: %d'%(len(val)))
    logger.info('Test size: %d'%(len(test)))
    
    # Build vocab
    # inputs_char.build_vocab(train.inputs_char, val.inputs_char, test.inputs_char)
    glove_vectors = GloVe(name='6B', dim=300)
    inputs_word.build_vocab(train.inputs_word, val.inputs_word, test.inputs_word, max_size=50000,
                        vectors=[glove_vectors])
    
    labels.build_vocab(train.labels)
    logger.info('Input vocab size:%d'%(len(inputs_word.vocab)))
    logger.info('Tagset size: %d'%(len(labels.vocab)))

    # Get iterators
    train_iter, val_iter, test_iter = data.BucketIterator.splits(
                            (train, val, test), batch_size=batch_size, 
                            device=torch.device("cuda:0" if torch.cuda.is_available() and use_cuda else "cpu"))
    train_iter.repeat = False
    val_iter.repeat = False

    # add embeddings
    embedding_size = inputs_word.vocab.vectors.shape[1]
    source_embedding = get_embedding(inputs_word.vocab, embedding_size)
    
    examples = train.examples[0:3] + val.examples[0:3] + test.examples[0:3]
    return {
        'task': 'conll2003.%s'%tag_type,
        'iters': (train_iter, val_iter, test_iter), 
        'vocabs': (inputs_word.vocab, labels.vocab),
        'word_field': inputs_word,
        'examples': examples,
        'embeddings': (source_embedding, None),
        'dummy_input': Variable(torch.zeros((batch_size, 42), dtype=torch.long))
        }

def extract_samples(samples: List[torchtext.data.example.Example]) -> ExampleList:
    result: ExampleList = []
    for example in samples:
        input_words: List[str] = example.inputs_word
        labels: List[str] = example.labels
        result.append(list(zip(input_words, labels)))
    return result


