from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import models.transformer.constants as constants
from models.transformer.layers import *

class Encoder(nn.Module):

    def __init__(self,
                n_enc_blocks=constants.DEFAULT_ENCODER_BLOCKS,
                n_head=constants.DEFAULT_NUMBER_OF_ATTENTION_HEADS,
                d_model=constants.DEFAULT_LAYER_SIZE,
                dropout_rate=constants.DEFAULT_MODEL_DROPOUT,
                pointwise_layer_size=constants.DEFAULT_DIMENSION_OF_PWFC_HIDDEN_LAYER,
                d_k=constants.DEFAULT_DIMENSION_OF_KEYQUERY_WEIGHTS,
                d_v=constants.DEFAULT_DIMENSION_OF_VALUE_WEIGHTS):
        """
        """
        super(Encoder, self).__init__()

        self.positional_encoding = PositionalEncoding(d_model)
        self.n_head = n_head
        self.encoder_blocks = []
        self.n_enc_blocks = n_enc_blocks
        self.d_model = d_model
        self.dropout_rate = dropout_rate
        self.pointwise_layer_size = pointwise_layer_size
        self.d_k = d_k
        self.d_v = d_v

        self._initialize_encoder_blocks()
        self.layer_norm = LayerNorm(d_model)
        

    def _initialize_encoder_blocks(self):
        for _ in range(self.n_enc_blocks):
            self.encoder_blocks.append(EncoderBlock(self.dropout_rate, self.pointwise_layer_size, self.d_model, self.d_k, self.d_v, self.n_head))

    def forward(self, x, mask=None):

        # create position embedding
        encoder_output = self.positional_encoding(x)

        # apply the forward pass for each encoding sub layer
        for enc_sub_layer in self.encoder_blocks:
            encoder_output = enc_sub_layer(encoder_output, mask)

        return encoder_output

    def print_model_graph(self, indentation: str = "") -> str:
        result = indentation + "- " + self.__str__() + "\n\n"
        result += indentation + "\t- " + self.positional_encoding.__str__() + "\n"
        for block in self.encoder_blocks:
            result += "\n" + block.print_model_graph(indentation + "\t")
        return result

    def __str__(self) -> str:
        return self.__class__.__name__ + "\Parameters\n" + self._get_parameters("")

    def _get_parameters(self, indentation: str) -> str:
        result = indentation + "\t# Blocks: {0}".format(self.n_enc_blocks)
        return result

        
class EncoderBlock(nn.Module):

    def __init__(self,
                dropout_rate,
                pointwise_layer_size,
                d_model,
                d_k,
                d_v,
                num_heads):
        """
        """
        super(EncoderBlock, self).__init__()

        self.dropout_rate = dropout_rate
        self.pointwise_layer_size = pointwise_layer_size
        self.d_model = d_model
        self.d_k = d_k
        self.d_v = d_v
        self.num_heads = num_heads
        self.self_attention_layer = MultiHeadedSelfAttentionLayer(d_k, d_v, d_model, num_heads, dropout_rate)
        self.feed_forward_layer = PointWiseFCLayer(d_model, pointwise_layer_size, dropout=self.dropout_rate)
        self.layer_norm = LayerNorm(d_model)

    def forward(self, x, mask=None):
        """Applies the forward pass on a transformer encoder layer.
        First, the input is put through the multi head attention.
        The result and the input are than added and normalized.
        Finally, this result is put through another feed forward network,
        followed by another norm layer.

        The output dimension is d_model = 512
        """
        # residual = x
        attentionResult = self.self_attention_layer(x, x, x, mask)
        # attentionResult = self.layer_norm.forward(attentionResult + residual)

        # residual = attentionResult

        fcResult = self.feed_forward_layer.forward(attentionResult)
        # fcResult = self.layer_norm.forward(fcResult + residual)

        return fcResult
        
    def __str__(self):
        return self.__class__.__name__

    def _get_parameters(self, indentation: str) -> str:
        result = indentation + "\tModel Size: {0}".format(self.d_model)
        return result

    def print_model_graph(self, indentation: str) -> str:
        result = indentation + "{\n"
        result += indentation + "- " + self.__str__() + ":\tParameters\n" + self._get_parameters(indentation + "\t") + "\n"
        result += self.self_attention_layer.print_model_graph(indentation + "\t") + "\n"
        result += self.feed_forward_layer.print_model_graph(indentation) + "\n"

        result += indentation + "- " + self.layer_norm.__str__() + "\n"
        result += indentation + "}\n"

        return result

if __name__ == '__main__':
    num_units = 512
    torch.manual_seed(42)
    # 10 words with a 100-lenght embedding
    inputs = Variable(torch.randn((100, 10)))

    # first 'layer'
    encoder = Encoder(2, 8, num_units, 0.1, 1024, 64, 64)
    print(encoder.print_model_graph())
    outputs = encoder(inputs)

    print(outputs)   