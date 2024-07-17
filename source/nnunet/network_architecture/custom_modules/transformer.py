'''
The implementation of the transformer block

Traditional self-attention mechanism with softmax
'''

import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F
from torch.nn.modules.dropout import Dropout
from torch.nn.modules.normalization import LayerNorm

import math
import copy
from typing import Optional

from .pos_embedding import AbsolutePosEncoding, \
                           LearnablePosEncoding, \
                           Conv3dPosEmbedding

def clones(module, N):
    """
    Produce N identical layers.
    Args:
        module: the model for copy
        N: the copy times
    """
    return torch.nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

def attention(query:Tensor, key:Tensor, value:Tensor, mask:Optional[Tensor]=None, dropout=None):
    """
    Compute the attention between q, k , v
    Input query_size:
        [batch_size, nhead, N, d_k]
    """
    d_model = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_model)
    if mask is not None:
        scores = scores.masked_fill(mask==0, -1e9)

    score_softmax = F.softmax(scores, dim=-1)

    if dropout is not None:
        score_softmax = dropout(score_softmax)
    
    out = torch.matmul(score_softmax, value)
    return out, score_softmax

def linear_attention(query:Tensor, key:Tensor, value:Tensor, mask:Optional[Tensor]=None, dropout=None):
    """
    Compute the linear attention between q, k , v
    Implementation of:
        https://arxiv.org/abs/1812.01243
    Input query_size:
        [batch_size, nhead, N, d_k]
    """
    d_model = query.size(-1)
    query = F.softmax(query, dim=-1) / math.sqrt(d_model)
    '''
    key = F.gelu(key)
    value = F.gelu(value)
    '''
    if mask is not None:
        key = key.masked_fill_(~mask, -1e9)
        value = value.masked_fill_(~mask, 0)

    key = F.softmax(key, dim=-2)
    context = torch.einsum('bhnd, bhne->bhde', key, value)

    if dropout is not None:
        score_softmax = dropout(query)

    out = torch.einsum('bhnd, bhde->bhne', query, context)

    return out, score_softmax

class MultihAttention(nn.Module):
    """
    Multihead attention implementation
    Args:
        d_model, the number of feature length of the input
        nhead: the number of heads for multihead self attention
        dropout: the dropout value
    Output:
        the result after multiheadattention, same size with input
    """
    def __init__(self, d_model:int, nhead:int, dropout:float):
        super(MultihAttention, self).__init__()
        assert d_model % nhead == 0, 'the dimension of feature should be devided by num head'
        self.d_model = d_model
        self.d_k = d_model // nhead
        self.nhead = nhead

        self.linears = clones(nn.Linear(d_model, d_model), N=4)
        self.attn = None
        self.dropout = Dropout(p=dropout)

    def forward(self, query:Tensor, key:Tensor, value:Tensor, 
                src_mask:Optional[Tensor]=None):
        
        if src_mask is not None:
            src_mask = src_mask.unsqueeze(1)

        n_batch = query.size(0)
        query, key, value = \
            [l(x).view(n_batch, -1, self.nhead, self.d_k).transpose(1, 2)
                for l, x in zip(self.linears, (query, key, value))]
        '''
        # Try linear attention here
        x, self.attn = attention(query=query, key=key, value=value, mask=src_mask,
                                 dropout=self.dropout)
        '''
        x, self.attn = linear_attention(query=query, key=key, value=value, mask=src_mask,
                                 dropout=self.dropout)
        x = x.transpose(1, 2).contiguous().view(n_batch, -1, self.nhead * self.d_k)
        return self.linears[-1](x)


class SelfAttentionLayer(nn.Module):
    """
    SelfAttention layer is made up of multihead self attention and feedforward network
    Args:
        d_model: the number of feature length of the input
        nhead: the number of heads for multihead self attention
        dim_feedforward: the dimension of the feed forward network
        dropout: the dropout value
        activation: the activation function of intermediate layer: relu or gelu
        layer_norm_eps: the eps value in layer normalization
    Output:
        The encoded feature using transformer encoder layer
    """
    def __init__(self, d_model:int, nhead:int, dim_feedforward:int, dropout:float=0.1,
                 activation="gelu", layer_norm_eps=1e-6):
        super(SelfAttentionLayer, self).__init__()
        self.d_model = d_model
        self.self_attn = MultihAttention(d_model, nhead, dropout=dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(p=dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.layer_norm1 = LayerNorm(d_model, eps=layer_norm_eps)
        # self.layer_norm1 = nn.InstanceNorm1d(d_model, eps=layer_norm_eps)
        self.layer_norm2 = LayerNorm(d_model, eps=layer_norm_eps)
        # self.layer_norm2 = nn.InstanceNorm1d(d_model, eps=layer_norm_eps)
        self.dropout1 = Dropout(p=dropout)
        self.dropout2 = Dropout(p=dropout)

        if activation == 'relu':
            self.activation = F.relu
        else:
            self.activation = F.gelu

    def forward(self, x, src_mask:Optional[Tensor]=None)->Tensor:
        x1 = self.self_attn(x, x, x, src_mask=src_mask)
        x = x + self.dropout1(x1)
        x = self.layer_norm1(x)

        x2 = self.linear2(self.dropout(self.activation(self.linear1(x))))
        x = x + self.dropout2(x2)
        x = self.layer_norm2(x)
        return x


class TransEncoder(nn.Module):
    """
    The transformer encoder structure
    Args:
        attn_layer: SelfAttentionLayer
        N: the repeat time
    """
    def __init__(self, attn_layer, N):
        super().__init__()
        self.layers = clones(attn_layer, N)
    
    def forward(self, x, mask:Optional[Tensor]=None):
        """
        repeat the self attention layer N times
        """
        for layer in self.layers:
                x = layer(x, mask)

        return x


class SelfAtten3DBlock(nn.Module):
    def __init__(self, in_dim, feature_length, d_model, nhead:int, dropout:float=0.3, N:int=8):
        '''
        Here is used to define the connection of skip encoded feature maps
        The transformer block will be only applied on the selected roi region
        return should be the same size with input
        Args:
            in_dim: input dimension from convolution
            feature_length: the feature length for pos_encoding
            d_model: the dimension for the model, use the bottle layer
            nhead: head for multihead attention
            N: attention repeated times
        Output:
            return the reshaped 3d convolutional block
        '''
        super().__init__()
        self.in_dim = in_dim
        self.d_model = d_model
        self.feature_length = feature_length
        self.linear_proj = nn.Linear(in_features=in_dim, out_features=d_model)
        self.linear_back_proj = nn.Linear(in_features=d_model, out_features=in_dim)
        self.pos_encode = AbsolutePosEncoding(max_length=feature_length, embedding_dim=d_model)
        attn_layer = SelfAttentionLayer(d_model=d_model, nhead=nhead, dim_feedforward=d_model, dropout=dropout)
        self.transformer = TransEncoder(attn_layer, N)

    def forward(self, x: torch.Tensor, mask:Optional[torch.Tensor]=None):
        '''
        x: Input x size
            [nbatch, channel, height, width, depth]
        Mask:
            [nbatch, 1, height, width, depth]
        '''
        nbatch, _, height, width, depth = x.shape
        x = x.flatten(start_dim=2).transpose(1, 2)
        x = self.linear_proj(x)
        x = self.pos_encode(x)
        if mask is not None:
            mask = mask.flatten(start_dim=2).transpose(1, 2)
        x = self.transformer(x, mask=mask)
        x = self.linear_back_proj(x)
        x = x.transpose(1, 2).reshape(nbatch, -1, height, width, depth)
        return x
