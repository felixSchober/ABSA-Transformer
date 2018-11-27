import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from models.transformer.layers import *
import models.transformer.constants as constants
from models.transformer.embeddings import Embeddings

class TransformerDecoder(nn.Module):
    """Some Information about Decoder"""
    def __init__(self,
                tgt_embeddings: nn.Embedding,
                d_vocab: int = None,
                num_decoder_blocks: int=constants.DEFAULT_DECODER_BLOCKS,
                n_head: int=constants.DEFAULT_NUMBER_OF_ATTENTION_HEADS,
                d_model: int=constants.DEFAULT_LAYER_SIZE,
                dropout_rate: float=constants.DEFAULT_MODEL_DROPOUT,
                pointwise_layer_size: int=constants.DEFAULT_DIMENSION_OF_PWFC_HIDDEN_LAYER,
                d_k: int=constants.DEFAULT_DIMENSION_OF_KEYQUERY_WEIGHTS,
                d_v: int=constants.DEFAULT_DIMENSION_OF_VALUE_WEIGHTS):
        super(TransformerDecoder, self).__init__()

        if tgt_embeddings is None:
            assert d_vocab is not None
            self.tgt_embeddings = Embeddings(d_model, d_vocab)
        else:
            self.tgt_embeddings = tgt_embeddings


        self.tgt_embeddings = tgt_embeddings
        self.decoder_blocks = []
        self.n_head = n_head
        self.num_decoder_blocks = num_decoder_blocks
        self.d_model = d_model
        self.dropout_rate = dropout_rate
        self.pointwise_layer_size = pointwise_layer_size
        self.d_k = d_k
        self.d_v = d_v

        self._initialize_decoder_blocks()
        self.layer_norm = LayerNorm(d_model)

    def _initialize_decoder_blocks(self):
        for _ in range(self.num_decoder_blocks):
            self.decoder_blocks.append(DecoderBlock(self.dropout_rate,
                                                    self.pointwise_layer_size,
                                                    self.d_model,
                                                    self.d_k,
                                                    self.d_v,
                                                    self.n_head))

    def forward(self, targets: torch.Tensor, encodings: torch.Tensor, source_mask: torch.Tensor, target_mask: torch.Tensor) -> torch.Tensor:
        """Applies forward pass on decoder to decode an encoding
        
        Arguments:
            targets {torch.Tensor} -- targets for the decoder
            encodings {torch.Tensor} -- result from an encoder
            source_mask {torch.Tensor} -- mask to apply on the source
            target_mask {torch.Tensor} -- mask to apply on the targets to prevent temporal peeking
        
        Returns:
            torch.Tensor -- result of forward pass
        """

        target_embeddings = self.tgt_embeddings(targets)

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
                d_model,
                d_k,
                d_v,
                num_heads):
        super(DecoderBlock, self).__init__()

        self.dropout_rate = dropout_rate
        self.pointwise_layer_size = pointwise_layer_size
        self.d_model = d_model
        self.d_k = d_k
        self.d_v = d_v
        self.num_heads = num_heads

        self.self_attention_layer = MultiHeadedSelfAttentionLayer(d_k, d_v, d_model, num_heads, dropout_rate)
        self.source_attention_layer = MultiHeadedSelfAttentionLayer(d_k, d_v, d_model, num_heads, dropout_rate)

        self.feed_forward_layer = PointWiseFCLayer(d_model, pointwise_layer_size, dropout=self.dropout_rate)
        self.layer_norm = LayerNorm(d_model)

    def forward(self, target_embeddings, encodings, source_mask, target_mask):
        x = target_embeddings
        x = self.self_attention_layer.forward(x, x, x, target_mask)
        x = self.source_attention_layer.forward(x, encodings, encodings, source_mask)
        x = self.feed_forward_layer(x)
        return x

def create_subsequent_mask(size: int) -> torch.Tensor:
    attention_shape = (1, size, size)
    mask = np.ones(attention_shape)
    mask = np.triu(mask, k=1).astype('uint8')
    return torch.from_numpy(mask) == 0
