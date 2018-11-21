from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from layers import *

class Encoder(nn.Module):

    def __init__(self, num_layers=6, layer_size=512, dropout=0.1):
        """
        """
        super(Encoder, self).__init__()

        self.positional_encoding = PositionalEncoding(layer_size)

        self.layers = []
        self.num_layers = num_layers
        self.layer_size = layer_size
        self.dropout = dropout
        for _ in range(num_layers):
            self.layers.append(EncoderBlock(layer_size))

    def forward(self, x):

        # create word embedding
        # TODO:
        w_emb = x

        # create position embedding
        encoder_output = self.positional_encoding(x)

        # apply the forward pass for each encoding sub layer
        for enc_sub_layer in self.layers:
            encoder_output = enc_sub_layer(encoder_output)

        return encoder_output

        
class EncoderBlock(nn.Module):

    def __init__(self, layer_size, dropout=0.1):
        """
        """
        super(EncoderBlock, self).__init__()

        self.dropout = dropout
        self.self_attention_layer = MultiHeadedSelfAttentionLayer()
        self.feed_forward_layer = PointWiseFCLayer(layer_size, dropout=self.dropout)
        self.layer_norm = LayerNorm(layer_size)

    def forward(self, x):
        """Applies the forward pass on a transformer encoder layer.
        First, the input is put through the multi head attention.
        The result and the input are than added and normalized.
        Finally, this result is put through another feed forward network,
        followed by another norm layer.

        The output dimension is d_model = 512
        """
        residual = x
        attentionResult = self.self_attention_layer(x, x, x)
        attentionResult = self.layer_norm.forward(attentionResult + residual)

        residual = attentionResult

        fcResult = self.feed_forward_layer.forward(attentionResult)
        fcResult = self.layer_norm.forward(fcResult + residual)

        return fcResult

if __name__ == '__main__':
    num_units = 512
    torch.manual_seed(42)
    # 10 words with a 100-lenght embedding
    inputs = Variable(torch.randn((100, 10)))

    # first 'layer'
    encoder = Encoder(2, num_units)
    outputs = encoder(inputs)

    print(outputs)   