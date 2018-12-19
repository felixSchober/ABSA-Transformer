import logging
from typing import Tuple, List, Dict, Optional, Union, Iterable
import pandas as pd
from data.conll import ExampleList
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm.autonotebook import tqdm
import torchtext

ExampleBatch = Tuple[torch.Tensor, torch.Tensor, List[str], List[str], torchtext.data.ReversibleField]
ExampleIterator = Iterable[ExampleBatch]

logger = logging.getLogger('prediction')


def print_samples(samples: ExampleList) -> None:
    for sample in samples:
        for word, label in sample:
            print('{} - {}'.format(word, label))
        print('\n#######################\n')

def iterate_with_sample_data(data_iterator: torchtext.data.Iterator, num_samples:int=5) -> ExampleIterator:
    assert num_samples > 0
    
    data_iterator.batch_size = 1
    data_iterator.init_epoch()

    for i, batch in enumerate(data_iterator):
        if i > num_samples:
            break
        x = batch.inputs_word
        y = batch.labels

        reversed_input = batch.dataset.fields['inputs_word'].reverse(x)
        reversed_label = batch.dataset.fields['labels'].reverse(y)

        yield (x, y, reversed_input, reversed_label, batch.dataset.fields['labels'])

def predict_samples(model: nn.Module, data_iterator: torchtext.data.Iterator, num_samples: int=5):
    with torch.no_grad:
        iterator = iterate_with_sample_data(data_iterator, num_samples)
        for x, y, sample_text, sample_label, label_reverser in iterator:
            prediction = model.predict(x, None)
            matches = count_matches(y, prediction)
            # try to reverse prediction (might not work if model is way off)
            try:
                prediction = label_reverser.reverse(prediction)
            except IndexError as err:
                prediction = ['?']
            except Exception as err:
                logger.exception('Could not reverse prediction. Unexpected error')
            yield(sample_text, sample_label, prediction, matches)

def count_matches(y, y_hat) -> int:
    return ((y == y_hat).sum()).item()

def predict_some_examples(model: nn.Module, iterator: ExampleIterator, num_samples: int=5):
    result = []
    for x, y, y_hat, num_matches in tqdm(predict_samples(model, iterator, num_samples), desc='Predicting'):
        line = []
        line.append(" ".join(x))
        line.append(" ".join(y))
        line.append(" ".join(y_hat))
        line.append(num_matches)
        result.append(line)
    return result

def predict_some_examples_to_df(model: nn.Module, iterator: ExampleIterator, num_samples: int=5):
    predictions = predict_some_examples(model, iterator, num_samples)
    return pd.DataFrame(predictions, columns=['Sentence', 'Targets', 'Predictions', '# Matches'])
