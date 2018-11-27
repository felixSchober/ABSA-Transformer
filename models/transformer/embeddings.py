import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class Embeddings(nn.Module):
    """Some Information about Embeddings"""
    def __init__(self, d_model: int, vocabulary_size: int):
        super(Embeddings, self).__init__()
        self.d_model = d_model
        self.vocabulary_size = vocabulary_size

        self.embedding_projection = nn.Embedding(vocabulary_size, d_model)

    def forward(self, x):
        return self.embedding_projection(x) * math.sqrt(self.d_model)