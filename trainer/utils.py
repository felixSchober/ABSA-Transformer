import torch

def create_padding_masks(targets: torch.Tensor, padd_class: int) -> torch.Tensor:
		input_mask = (targets != padd_class).unsqueeze(-2)
		return input_mask