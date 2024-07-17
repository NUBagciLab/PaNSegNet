'''
Implement of positional encoding
Including:
    Absolute positional encoding
    Learnable positional encoding
    Convolution-based positional encoding
'''

from turtle import forward
import torch
import torch.nn as nn

class AbsolutePosEncoding(nn.Module):
    def __init__(self, max_length:int, embedding_dim:int):
        super().__init__()
        self.max_length = max_length
        self.embedding_dim = embedding_dim
        pe = torch.zeros((max_length, embedding_dim))
        position = torch.arange(0, max_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, embedding_dim, 2).float() *
            (-torch.log(torch.tensor(10000.0)) / embedding_dim)
        )
        pe[:, 0::2] = torch.sin(position*div_term)
        pe[:, 1::2] = torch.cos(position*div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        x = x + self.pe
        return x


class LearnablePosEncoding(nn.Module):
    def __init__(self, max_length:int, embedding_dim:int):
        super().__init__()
        self.max_length = max_length
        self.embedding_dim = embedding_dim
        self.pe = nn.Parameter(torch.zeros(size=(1, self.max_length, self.embedding_dim)))

    def forward(self, x):
        return x + self.pe



class Conv3dPosEmbedding(nn.Module):
    '''
    Positinal Encoding Generator using 3d convolution
    Args:
        dim: the input feature dimension
        dropout: the dropout ratio
        emb_kernel: the kernel size of convolution
            padding_size = emb_kernel // 2
    '''
    def __init__(self, dim, dropout:float, emb_kernel:int=3):
        super(Conv3dPosEmbedding, self).__init__()
        self.proj = nn.Conv3d(in_channels=dim, out_channels=dim, kernel_size=emb_kernel,
                              stride=1, padding=emb_kernel//2, groups=dim)
        
        self.dropout = nn.Dropout3d(p=dropout)
    
    def forward(self, x):
        """
        Args:
            Input: 
                size, [batch, channels, heights, widths, depths]
            Output:
                size is same with Input
        """
        pos_enc = self.proj(x)
        x = x + pos_enc
        return self.dropout(x)


class Conv2dPosEmbedding(nn.Module):
    '''
    Positinal Encoding Generator using convolution
    Args:
        dim: the input feature dimension
        dropout: the dropout ratio
        emb_kernel: the kernel size of convolution
            padding_size = emb_kernel // 2
    '''
    def __init__(self, dim, dropout:float, emb_kernel:int=3):
        super(Conv2dPosEmbedding, self).__init__()
        self.proj = nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=emb_kernel,
                              stride=1, padding=emb_kernel//2, groups=dim)
        
        self.dropout = nn.Dropout2d(p=dropout)
    
    def forward(self, x):
        """
        Args:
            Input: 
                size, [batch, channels, heights, widths, depths]
            Output:
                size is same with Input
        """
        pos_enc = self.proj(x)
        x = x + pos_enc
        return self.dropout(x)
