import torch
import torch.nn as nn
import torch.nn.functional as F

from layers import *
import constants

class Decoder(nn.Module):
    """Some Information about Decoder"""
    def __init__(self,
                num_decoder_blocks=constants.DEFAULT_DECODER_BLOCKS,
                num_attention_heads=constants.DEFAULT_NUMBER_OF_ATTENTION_HEADS,
                model_size=constants.DEFAULT_LAYER_SIZE,
                dropout_rate=constants.DEFAULT_MODEL_DROPOUT,
                pointwise_layer_size=constants.DEFAULT_DIMENSION_OF_PWFC_HIDDEN_LAYER,
                key_query_dimension=constants.DEFAULT_DIMENSION_OF_KEYQUERY_WEIGHTS,
                value_dimension=constants.DEFAULT_DIMENSION_OF_VALUE_WEIGHTS):
        super(Decoder, self).__init__()

        self.decoder_blocks = []
        self.num_attention_heads = num_attention_heads
        self.num_decoder_blocks = num_decoder_blocks
        self.model_size = model_size
        self.dropout_rate = dropout_rate
        self.pointwise_layer_size = pointwise_layer_size
        self.key_query_dimension = key_query_dimension
        self.value_dimension = value_dimension

        self._initialize_decoder_blocks()
        self.layer_norm = LayerNorm(model_size)

    def _initialize_decoder_blocks(self):
        for _ in range(self.num_encoder_blocks):
            self.decoder_blocks.append(DecoderBlock(self.dropout_rate,
                                                    self.pointwise_layer_size,
                                                    self.model_size,
                                                    self.key_query_dimension,
                                                    self.value_dimension,
                                                    self.num_attention_heads))

    def forward(self, target_embeddings, encodings, source_mask, target_mask):
        for decoder_block in self.decoder_blocks:
            target_embeddings = decoder_block(target_embeddings,
                                                encodings,
                                                source_mask,
                                                target_mask)
        return target_embeddings


class DecoderBlock(nn.Module):
    """Some Information about DecoderBlock"""
    def __init__(self,
                dropout_rate,
                pointwise_layer_size,
                model_size,
                key_query_dimension,
                value_dimension,
                num_heads):
        super(DecoderBlock, self).__init__()

        self.dropout_rate = dropout_rate
        self.pointwise_layer_size = pointwise_layer_size
        self.model_size = model_size
        self.key_query_dimension = key_query_dimension
        self.value_dimension = value_dimension
        self.num_heads = num_heads

        self.self_attention_layer = MultiHeadedSelfAttentionLayer(key_query_dimension, value_dimension, model_size, num_heads, dropout_rate)
        self.source_attention_layer = MultiHeadedSelfAttentionLayer(key_query_dimension, value_dimension, model_size, num_heads, dropout_rate)

        self.feed_forward_layer = PointWiseFCLayer(model_size, pointwise_layer_size, dropout=self.dropout_rate)
        self.layer_norm = LayerNorm(model_size)

    def forward(self, target_embeddings, encodings, source_mask, target_mask):
        x = target_embeddings
        x = self.self_attention_layer.forward(x, x, x, target_mask)
        x = self.source_attention_layer.forward(x, encodings, encodings, source_mask)
        x = self.feed_forward_layer(x)
        return x
