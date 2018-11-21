import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import copy

import constants

def clone_layer(layer: nn.Module, N: int):
    """Produces N identitcal layers
    """
    return nn.ModuleList([copy.deepcopy(layer) for _ in range(N)])


class PointWiseFCLayer(nn.Module):

    def __init__(self, d_input=constants.DEFAULT_DIMENSION_OF_MODEL, d_layer=constants.DEFAULT_DIMENSION_OF_PWFC_HIDDEN_LAYER, dropout=constants.DEFAULT_MODEL_DROPOUT) -> None:
        super(PointWiseFCLayer, self).__init__()

        self.d_input = d_input
        self.d_layer = d_layer
        self.p_dropout = dropout

        self.w_1 = nn.Conv1d(d_input, d_layer, 1) 
        self.w_2 = nn.Conv1d(d_layer, d_input, 1)      # output dimension = input dimension 
        self.dropout = nn.Dropout(self.p_dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        result = self.w_2(F.relu(self.w_1(x)))
        result = self.dropout(x)
        return result

class ScaledDotProductAttentionLayer(nn.Module):

    def __init__(self,
                d_k=constants.DEFAULT_DIMENSION_OF_KEYQUERY_WEIGHTS,
                d_v=constants.DEFAULT_DIMENSION_OF_VALUE_WEIGHTS,
                dropout=constants.DEFAULT_MODEL_DROPOUT) -> None:
        """
        d_k: dimensionality of the query and key vectors    (default 64)
        d_v: dimensionality of the value vector             (default 64)
        dropout                                             (default 0.1)
        """
        super(ScaledDotProductAttentionLayer, self).__init__()

        self.d_k = d_k
        self.d_v = d_v
        self.gradientStabilizer = math.sqrt(self.d_k)

        # for a isolated scaled dot-product attention we would use this.
        # However, for multi-head attention we project a single v, k, q matrix
        # using multiple linear layers (one for each matrix and attention head) -> 3 * number of heads layers
        # self.q_matrix = nn.Sequential(nn.Linear(self.d_k, self.d_k), nn.ReLU())
        # self.k_matrix = nn.Sequential(nn.Linear(self.d_k, self.d_k), nn.ReLU())
        # self.v_matrix = nn.Sequential(nn.Linear(self.d_v, self.d_v), nn.ReLU())

        if dropout is not None:
            self.dropout = nn.Dropout(dropout)
        else:
            self.dropout = None

    def _self_attention(self,
                        x: torch.Tensor,
                        w_query: torch.Tensor,
                        w_key: torch.Tensor,
                        w_value: torch.Tensor,
                        mask: torch.Tensor =None,
                        dropout: torch.Tensor=None,
                        eps=1e-6) -> (torch.Tensor, torch.Tensor):
        """Calculates attention for one head.
        This method is the 'Scaled Dot-Product Attention mechanism from the paper.
        Corresponds to Figure 2 left
        x: input sentence (list of word embeddings of sentence)     [embedding_size, num_words] (512, n)
        w_query: query matrix                                       [d_k, embedding_size]       (64, 512)
        w_key: key matrix                                           [d_k, embedding_size]       (64, 512)
        w_value: value matrix                                       [d_v, embedding_size]       (64, 512)
        mask: mask to prevent leftward information flow             [num_words, num_words]      (n, n)
        dropout: dropout layer
        eps: epsilon value
        """

        # Step 1:   Multiply input (matrix of word embeddings of sentence) with query, key and value matrixes
        #           Result is a query, key and value vector for each word
        x_w_query = x * w_query                                 #   [d_k, num_words]            (64, n)
        x_w_key = x * w_key                                     #   [d_k, num_words]            (64, n)
        x_w_value = x * w_value                                 #   [d_v, num_words]            (64, n)

        # Step 2:   For each word in input sentence, calculate a score against the current word
        #               for w_1 in sentence
        #                   for w_2 in sentence
        #                       query_vector(w_1) * key_vector(w_2)

        #           Divide by the square root of the query/key/value matrix sizes (default 8)
        scores = torch.matmul(x_w_query * torch.t(x_w_key)) / self.gradientStabilizer   # [num_words, num_words]       

        # Step 3:   To prevent leftward information flow, we need to mask out words that the attention
        #           head is not 'allowed' to see
        if mask is not None:
            scores = scores.masked_fill(mask == 0, eps)

        # Step 4:   Create softmax of scores
        attention = F.softmax(scores, dim=-1)  # [num_words, num_words]     
        
        if dropout is not None:
            attention = dropout(attention)

        # Step 5:   multiply each score with their coresponding softmax score
        #           sum up all the scores -> output for one word      
        result = torch.matmul(attention, x_w_value)    # [d_v, num_words]      (64, n)
        return result, attention

    def _self_attention_multi_head(self,
                        w_query: torch.Tensor,
                        w_key: torch.Tensor,
                        w_value: torch.Tensor,
                        mask: torch.Tensor =None,
                        dropout: torch.Tensor=None,
                        eps=1e-6):
        """Calculates attention for multiple heads.
        This method is the 'Scaled Dot-Product Attention mechanism from the paper.
        Corresponds to Figure 2 left
        w_query: query matrix                                       [d_k, embedding_size, h]       (64, 512, 8)
        w_key: key matrix                                           [d_k, embedding_size, h]       (64, 512, 8)
        w_value: value matrix                                       [d_v, embedding_size, h]       (64, 512, 8)
        mask: mask to prevent leftward information flow             [num_words, num_words]         (n, n)
        dropout: dropout layer
        eps: epsilon value
        """

        # Step 1:   Multiply input (matrix of word embeddings of sentence) with query, key and value matrixes
        #           Result is a query, key and value vector for each word
        # **Already done**

        # Step 2:   For each word in input sentence, calculate a score against the current word
        #               for w_1 in sentence
        #                   for w_2 in sentence
        #                       query_vector(w_1) * key_vector(w_2)

        #           Divide by the square root of the query/key/value matrix sizes (default 8)
        # bmm = batch matrix multiplication
        scores = torch.bmm(w_query * torch.t(w_key)) / self.gradientStabilizer   # [num_words, num_words, h]       

        # Step 3:   To prevent leftward information flow, we need to mask out words that the attention
        #           head is not 'allowed' to see
        if mask is not None:
            scores = scores.masked_fill(mask == 0, eps)

        # Step 4:   Create softmax of scores
        # TODO: check dimension
        attention = F.softmax(scores, dim=-1)  # [num_words, num_words, h]     
        
        if dropout is not None:
            attention = dropout(attention)

        # Step 5:   multiply each score with their coresponding softmax score
        #           sum up all the scores -> output for one word      
        result = torch.bmm(attention, w_value)    # [d_v, num_words]      (64, n)
        return result, attention
        

    # def forward(self, x, q_matrix, k_matrix, v_matrix):
    #     return self._self_attention(x, q_matrix, k_matrix, v_matrix, None, self.dropout)

    def forward(self, q_matrix, k_matrix, v_matrix, mask):
        return self._self_attention_multi_head(q_matrix, k_matrix, v_matrix, mask, self.dropout)


class MultiHeadedSelfAttentionLayer(nn.Module):

    def __init__(self,
                d_k=constants.DEFAULT_DIMENSION_OF_KEYQUERY_WEIGHTS,
                d_v=constants.DEFAULT_DIMENSION_OF_VALUE_WEIGHTS,
                d_model=constants.DEFAULT_DIMENSION_OF_MODEL,
                h=constants.DEFAULT_NUMBER_OF_ATTENTION_HEADS,
                use_linear=True) -> None:
        """
        d_k: dimensionality of the query and key vectors
        d_v: dimensionality of the value vector
        h: number of attention heads
        """
        super(MultiHeadedSelfAttentionLayer, self).__init__()

        assert d_model % h == 0
        assert d_k * h == d_model

        self.d_model = d_model
        self.d_k = d_k
        self.d_v = d_v
        self.h = h

        # query, key, value projections
        # those matrices are used to transform the query, key and value matrix for each attention head
        # During forward pass these projection matrices are split so that they can be applied for each head
        # in the paper they are called W^Q, W^k and W^V
        self.query_projections = nn.Sequential(nn.Linear(self.d_model, self.d_k * self.h), nn.ReLU())
        self.key_projections = nn.Sequential(nn.Linear(self.d_model, self.d_k * self.h), nn.ReLU())
        self.value_projections = nn.Sequential(nn.Linear(self.d_model, self.d_v * self.h), nn.ReLU())

        # one 'attention_layer' is sufficient even if the model uses multiple heads since the layer
        # only performs a forward pass without any learned parameters
        self.attention_layer = ScaledDotProductAttentionLayer()

        self.layer_norm = LayerNorm(self.d_model)

        # after concatinating the attention output of the heads this last matrix is used to project the output
        # back to the original input size so that the output of this layer can be used again for the next layer
        # The input of this layer is the output of the forward pass of head attention head, multiplied by the number of heads
        # The output should be the model dimension again, so that the input dimension of the layer 
        # input_MultiHeadedSelfAttentionLayer = output_MultiHeadedSelfAttentionLayer
        self.w_0 = nn.Sequential(nn.Linear(self.h * self.d_v, self.d_model), nn.ReLU())

        self._initialize_layers()


    def _initialize_layers(self, normal: bool = True):
        if (normal):
            nn.init.normal(self.query_projections, mean=0, std=np.sqrt(2.0 / (self.d_model + self.d_k)))
            nn.init.normal(self.key_projections, mean=0, std=np.sqrt(2.0 / (self.d_model + self.d_k)))
            nn.init.normal(self.value_projections, mean=0, std=np.sqrt(2.0 / (self.d_model + self.d_v)))

        nn.init.xavier_normal_(self.w_0.weight)


    def _create_projections(self):
        None

    def _concatenate_attentions(self, scores: torch.Tensor, nbatches: int) -> torch.Tensor:
        # TODO: what is nbatches
        return scores.transpose(1, 2).contiguous().view(nbatches, -1, self.h * self.d_k)

    def forward(self, x_queries: torch.Tensor, x_keys: torch.Tensor, x_values: torch.Tensor, mask: torch.Tensor=None) -> torch.Tensor:

        # residual used as depicted in fig. 1
        residual = x_queries

        # project key, query, value for each head using the linear layers
        Q = self.query_projections(x_queries)
        K = self.key_projections(x_keys)
        V = self.value_projections(x_values)

        # split to head input dimensions
        # TODO: check dimensions
        sz_b, len_q, _ = x_queries.size()
        sz_b, len_k, _ = x_keys.size()
        sz_b, len_v, _ = x_values.size()

        Q = Q.view(sz_b, len_q, self.h, self.d_k)
        K = K.view(sz_b, len_k, self.h, self.d_k)
        V = V.view(sz_b, len_v, self.h, self.d_v)

        # padd
        # TODO: check what this does
        Q = Q.permute(2, 0, 1, 3).contiguous().view(-1, len_q, self.d_k) # (n*b) x lq x dk
        K = K.permute(2, 0, 1, 3).contiguous().view(-1, len_k, self.d_k) # (n*b) x lk x dk
        V = V.permute(2, 0, 1, 3).contiguous().view(-1, len_v, self.d_v) # (n*b) x lv x dv

        # apply mask
        mask = mask.repeat(self.h, 1, 1)

        # perform forward pass of individual heads
        result, attention = self.attention_layer.forward(Q, K, V, mask=mask)

        # prepare for merge by concatenating heads
        # TODO: check
        result = result.view(self.h, sz_b, len_q, self.d_v)
        result = result.permute(1, 2, 0, 3).contiguous().view(sz_b, len_q, -1) # b x lq x (n*dv)

        result = self.w_0(result)

        result = self.dropout(result)

        # add residual again
        result = self.layer_norm(result + residual)

        return result, attention



class LayerNorm(nn.Module):

    def __init__(self, features, eps=1e-6) -> None:
        """Applies layer normalization from Jimmy Lei Ba et al. Layer normalization
        """
        super(LayerNorm, self).__init__()
        self.gamma = nn.Parameter(torch.ones(features))
        self.beta = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.gamma * (x - mean) / (std + self.eps) + self.beta



