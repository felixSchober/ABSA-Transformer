import torch
import torch.nn as nn
import torch.nn.functional as F

from models.transformer.encoder import TransformerEncoder
from models.softmax_output import SoftmaxOutputLayer

class TransformerTagger(nn.Module):

    encoder: TransformerEncoder
    taggingLayer: SoftmaxOutputLayer

    def __init__(self, transformerEncoder: TransformerEncoder, taggingLayer: SoftmaxOutputLayer):
        super(TransformerTagger, self).__init__()

        self.encoder = transformerEncoder
        self.taggingLayer = taggingLayer

    def forward(self, x: torch.Tensor, *args) -> torch.Tensor:
        result = self.encoder(x, *args)
        return self.taggingLayer(result)

    def predict(self, x: torch.Tensor, *args):
        result = self.encoder(x, *args)
        return self.taggingLayer.predict(x)
