import torch
import torch.nn as nn
import torch.nn.functional as F

import constants
from encoder import Encoder
from decoder import Decoder

class GoogleTransformer(nn.Module):

    def __init__(self,
                vocabulary_size,
                num_encoder_blocks=constants.DEFAULT_ENCODER_BLOCKS,
                num_attention_heads=constants.DEFAULT_NUMBER_OF_ATTENTION_HEADS,
                model_size=constants.DEFAULT_LAYER_SIZE,
                dropout_rate=constants.DEFAULT_MODEL_DROPOUT,
                pointwise_layer_size=constants.DEFAULT_DIMENSION_OF_PWFC_HIDDEN_LAYER,
                key_query_dimension=constants.DEFAULT_DIMENSION_OF_KEYQUERY_WEIGHTS,
                value_dimension=constants.DEFAULT_DIMENSION_OF_VALUE_WEIGHTS):
        super(GoogleTransformer, self).__init__()

        self.encoder = Encoder()
        self.decoder = Decoder()

        self.source_embeddings = None
        self.target_embeddings = None
        self.generator = TransformerTargetGenerator(model_size, vocabulary_size)

    def forward(self, source: torch.Tensor, targets: torch.Tensor, source_mask: torch.Tensor, target_mask: torch.Tensor) -> torch.Tensor:
        encodingResult = self.encode(source, source_mask)
        decodingResult = self.decode(encodingResult, source_mask, target_mask, targets)
        return decodingResult
        

    def encode(self, x: torch.Tensor, source_mask: torch.Tensor) -> torch.Tensor:
        input_embeddings = self.source_embeddings(x)
        return self.encoder.forward(input_embeddings, source_mask)

    def decode(self, encodings: torch.Tensor, source_mask: torch.Tensor, target_mask, targets: torch.Tensor) -> torch.Tensor:
        target_embeddings = self.target_embeddings(targets)
        results = self.decoder(target_embeddings, encodings, source_mask, target_mask)
        return results


class TransformerTargetGenerator(nn.Module):
    """Some Information about TransformerTargetGenerator"""
    def __init__(self, model_size: int, vocabulary_size: int):
        super(TransformerTargetGenerator, self).__init__()
        self.model_size = model_size
        self.vocabulary_size = vocabulary_size
        self.result_projection = nn.Linear(self.model_size, self.vocabulary_size)

    def forward(self, x):
        result = self.result_projection(x)
        return F.log_softmax(result, dim=-1)

    def _get_parameters(self, indentation: str) -> str:
        return indentation + "\tVocabulary Size: {0}\n".format(self.vocabulary_size)

    def print_model_graph(self, indentation: str) -> str:
        return indentation + "- " + self.__str__() + ": - Parameters\n" + self._get_parameters(indentation + "\t") + "\n"

    def __str__(self) -> str:
        return self.__class__.__name__