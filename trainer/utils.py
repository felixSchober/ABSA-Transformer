import torch

METRIC_F1 = 'f1'
METRIC_LOSS = 'loss'
METRIC_PRECISSION = 'precission'
METRIC_RECALL = 'recall'
METRIC_ACCURACY = 'accuracy'

ITERATOR_TRAIN = 'train'
ITERATOR_VALIDATION = 'validation'
ITERATOR_TEST = 'test'


def create_padding_masks(targets: torch.Tensor, padd_class: int) -> torch.Tensor:
		input_mask = (targets != padd_class).unsqueeze(-2)
		return input_mask