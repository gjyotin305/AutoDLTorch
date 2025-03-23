from .utils import FeedForwardNetwork
from ..attention.utils import (
    MultiHeadAttention
)
import torch
import torch.nn as nn

class EncoderLayer(nn.Module):
    def __init__(self, d_model, d_hidden, n_head, drop_prob):
        super(EncoderLayer, self).__init__()
        self.mha = MultiHeadAttention(d_model=d_model, num_heads=n_head)
        self.norm1 = nn.LayerNorm(d_model)
        self.drop1 = nn.Dropout(drop_prob)

        self.ffn = FeedForwardNetwork(
            d_model=d_model, 
            d_hidden=d_hidden, 
            drop_prob=drop_prob
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.drop2 = nn.Dropout(drop_prob)
    
    def forward(self, x, mask=None):
        id_x = x
        out = self.mha(x, mask)
        
        out = self.drop1(out)
        out = self.norm1(out+id_x)
        
        id_x = out
        out = self.ffn(out)

        out = self.drop2(out)
        out = self.norm2(out+id_x)

        return out
