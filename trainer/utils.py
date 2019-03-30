import torch

METRIC_MACRO_F1 = 'macro f1'
METRIC_MICRO_F1 = 'micro f1'
METRIC_F1 = 'f1'

METRIC_PRECISSION = 'precission'
METRIC_RECALL = 'recall'

METRIC_MACRO_PRECISSION = 'macro precission'
METRIC_MACRO_RECALL = 'macro recall'


METRIC_ACCURACY = 'accuracy'


METRIC_LOSS = 'loss'


ITERATOR_TRAIN = 'train'
ITERATOR_VALIDATION = 'validation'
ITERATOR_TEST = 'test'


def create_padding_masks(targets: torch.Tensor, padd_class: int) -> torch.Tensor:
		input_mask = (targets != padd_class).unsqueeze(-2)
		return input_mask