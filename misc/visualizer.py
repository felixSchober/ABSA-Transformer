from data.conll import ExampleList, ExampleIterator, iterate_with_sample_data
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm.autonotebook import tqdm


def print_samples(samples: ExampleList) -> None:
    for sample in samples:
        for word, label in sample:
            print('{} - {}'.format(word, label))
        print('\n#######################\n')

def print_samples_with_prediction(model: nn.Module, iterator: ExampleIterator, num_samples: int=5):
    #with torch.no_grad:
        for x, y, sample_text, sample_label in tqdm(iterator, leave=False):
            prediction = model.predict(x, None)
            print(prediction)
