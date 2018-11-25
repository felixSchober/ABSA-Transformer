import math
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import *

import constants
from encoder import Encoder
from decoder import Decoder

class GoogleTransformer(nn.Module):

    def _init_loggers(self):
        # this logger will not print to the console. Only to the file.
        self.logger = logging.getLogger(__name__)

        # this logger will both print to the console as well as the file
        self.logger_prediction = logging.getLogger('prediction')

    def __init__(self,
                initialize_xavier,
                src_vocab_size: int,
                tgt_vocab_size: int,
                d_word_vec: int,
                n_enc_blocks=constants.DEFAULT_ENCODER_BLOCKS,
                n_head=constants.DEFAULT_NUMBER_OF_ATTENTION_HEADS,
                d_model=constants.DEFAULT_LAYER_SIZE,
                dropout_rate=constants.DEFAULT_MODEL_DROPOUT,
                pointwise_layer_size=constants.DEFAULT_DIMENSION_OF_PWFC_HIDDEN_LAYER,
                d_k=constants.DEFAULT_DIMENSION_OF_KEYQUERY_WEIGHTS,
                d_v=constants.DEFAULT_DIMENSION_OF_VALUE_WEIGHTS):
        """Initializes a Transformer Model
        
        Arguments:
            initialize_xavier {boolean} -- [description]
            src_vocab_size {int} -- Size / Length of the source vocabulary (how many tokens)
            tgt_vocab_size {[type]} -- Size / Length of the target vocabulary (how many tokens)
            d_word_vec {[type]} -- Dimension of word vectors (default: 512)
        
        Keyword Arguments:
            n_enc_blocks {[type]} -- [description] (default: {constants.DEFAULT_ENCODER_BLOCKS})
            n_head {[type]} -- [description] (default: {constants.DEFAULT_NUMBER_OF_ATTENTION_HEADS})
            d_model {[type]} -- [description] (default: {constants.DEFAULT_LAYER_SIZE})
            dropout_rate {[type]} -- [description] (default: {constants.DEFAULT_MODEL_DROPOUT})
            pointwise_layer_size {[type]} -- [description] (default: {constants.DEFAULT_DIMENSION_OF_PWFC_HIDDEN_LAYER})
            d_k {[type]} -- [description] (default: {constants.DEFAULT_DIMENSION_OF_KEYQUERY_WEIGHTS})
            d_v {[type]} -- [description] (default: {constants.DEFAULT_DIMENSION_OF_VALUE_WEIGHTS})
        """

        super(GoogleTransformer, self).__init__()

        assert d_model == d_word_vec, \
        'To facilitate the residual connections, \
         the dimensions of all module outputs shall be the same. Input (Word Embedding) = Output of Encoder Layer'

        self._init_loggers()

        self.encoder = Encoder()
        self.decoder = Decoder()

        # generate a last layer that projects from the last output of the decoder layer
        # with size d_model to the size of the target vocabulary
        self.decoder_to_tgt_vocabulary = nn.Linear(d_model, tgt_vocab_size, bias=False)

        # initialize embeddings
        self.source_embeddings = Embeddings(d_model, src_vocab_size)
        self.target_embeddings = Embeddings(d_model, tgt_vocab_size)
        self.generator = TransformerTargetGenerator(d_model, src_vocab_size)

        if (initialize_xavier):
            for p in self.parameters():
                if p.dim() > 1:
                    nn.init.xavier_uniform_(p)

    def forward(self, source: torch.Tensor, targets: torch.Tensor, source_mask: torch.Tensor, target_mask: torch.Tensor) -> torch.Tensor:
        encodingResult = self.encode(source, source_mask)
        decodingResult = self.decode(encodingResult, source_mask, target_mask, targets)

        # project to final vocabulary size
        result = self.decoder_to_tgt_vocabulary(decodingResult) # result now has size [batch_size, longest_sequence_lenght, vocabulary_size]

        # TODO: what does this do
        return result.view(-1, result.size(2))

    def encode(self, x: torch.Tensor, source_mask: torch.Tensor) -> torch.Tensor:
        input_embeddings = self.source_embeddings(x)
        return self.encoder.forward(input_embeddings, source_mask)

    def decode(self, encodings: torch.Tensor, source_mask: torch.Tensor, target_mask, targets: torch.Tensor) -> torch.Tensor:
        target_embeddings = self.target_embeddings(targets)
        results = self.decoder(target_embeddings, encodings, source_mask, target_mask)
        return results


class TransformerTargetGenerator(nn.Module):
    """Some Information about TransformerTargetGenerator"""
    def __init__(self, d_model: int, vocabulary_size: int):
        super(TransformerTargetGenerator, self).__init__()
        self.d_model = d_model
        self.vocabulary_size = vocabulary_size
        self.result_projection = nn.Linear(self.d_model, self.vocabulary_size)

    def forward(self, x):
        result = self.result_projection(x)
        return F.log_softmax(result, dim=-1)

    def _get_parameters(self, indentation: str) -> str:
        return indentation + "\tVocabulary Size: {0}\n".format(self.vocabulary_size)

    def print_model_graph(self, indentation: str) -> str:
        return indentation + "- " + self.__str__() + ": - Parameters\n" + self._get_parameters(indentation + "\t") + "\n"

    def __str__(self) -> str:
        return self.__class__.__name__

class Embeddings(nn.Module):
    """Some Information about Embeddings"""
    def __init__(self, d_model: int, vocabulary_size: int):
        super(Embeddings, self).__init__()
        self.d_model = d_model
        self.vocabulary_size = vocabulary_size

        self.embedding_projection = nn.Embedding(vocabulary_size, d_model)

    def forward(self, x):
        return self.embedding_projection(x) * math.sqrt(self.d_model)


# testing transformer model forward pass
if __name__ == '__main__':
    num_units = 512
    torch.manual_seed(42)
    # 10 words with a 100-lenght embedding
    transformer = GoogleTransformer(True, 3, 3, 512, 2, 2, 512, 0.1)

    input_idices = [[0, 1, 0, 0], [1, 0, 1, 1]]
    inputs = Variable(torch.tensor(input_idices, dtype=torch.long))
    targets = inputs.clone()
    result = transformer.forward(inputs, targets, None, None)
    print(result)
