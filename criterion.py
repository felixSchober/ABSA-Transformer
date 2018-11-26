import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossEntropyLoss2d(nn.Module):
    """
    taken from https://github.com/bodokaiser/piwise/tree/master/piwise
    """
    def __init__(self, n_classes, weight=None):
        super().__init__()
        self.n_classes = n_classes
        self.loss = nn.NLLLoss2d(weight)

    def forward(self, outputs, targets):
        return self.loss(F.log_softmax(outputs), targets[0])

    @property
    def output_frame_size(self):
        return self.n_classes


class NllLoss(nn.Module):
    def __init__(self, output_size):
        super().__init__()
        self.output_size = output_size

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        log_probs = F.log_softmax(logits, -1)
        return F.nll_loss(log_probs.view(-1, self.output_size), targets.view(-1))
    



class MSELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss = nn.MSELoss()

    def forward(self, outputs, targets):
        return self.loss(outputs, targets)

    @property
    def output_frame_size(self):
        return 1