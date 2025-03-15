import torch.nn as nn
from ....attention.utils import (
    MultiHeadAttention
)
from ...utils import (
    FeedForwardNetwork
)

class DecoderLayer(nn.Module):
    def __init__(self, d_model, d_hidden, n_head, drop_prob):
        super(DecoderLayer, self).__init__()
        self.mha = MultiHeadAttention(d_model=d_model, num_heads=n_head)
        self.norm1 = nn.LayerNorm(d_model)
        self.drop1 = nn.Dropout(drop_prob)

        self.enc_dec_attention = MultiHeadAttention(
            d_model=d_model, num_heads=n_head
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.drop2 = nn.Dropout(drop_prob)

        self.ffn = FeedForwardNetwork(
            d_model=d_model, 
            d_hidden=d_hidden, 
            drop_prob=drop_prob
        )
        self.norm3 = nn.LayerNorm(d_model)
        self.drop3 = nn.Dropout(drop_prob)

    def forward(self, x_dec, x_enc, trg_mask, src_mask):
        id_x = x_dec
        out = self.mha(x_dec, trg_mask)

        out = self.drop1(out)
        out = self.norm1(out+id_x)

        if x_enc is not None:
            id_x = out
            out = self.enc_dec_attention(out, src_mask)

            out = self.drop2(out)
            out = self.norm2(out+id_x)
        
        id_x = out
        out  = self.ffn(out)

        out = self.drop3(out)
        out = self.norm3(out+id_x)

        return out