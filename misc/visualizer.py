import logging
import pandas as pd
from data.conll import ExampleList, ExampleIterator, iterate_with_sample_data
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm.autonotebook import tqdm


logger = logging.getLogger('prediction')


def print_samples(samples: ExampleList) -> None:
    for sample in samples:
        for word, label in sample:
            print('{} - {}'.format(word, label))
        print('\n#######################\n')

def predict_samples(model: nn.Module, iterator: ExampleIterator, num_samples: int=5):
    #with torch.no_grad:
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
