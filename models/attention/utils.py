import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

class SelfAttention(nn.Module):
    def __init__(self, d_model):
        super(SelfAttention, self).__init__()
        self.d_model = d_model

        self.qkv_proj = nn.Linear(self.d_model, 3*self.d_model)
    
    def forward(self, x):
        qkv = self.qkv_proj(x)
        q, k, v = qkv.chunk(3, dim=-1)

        scaled_sims = q@(k.transpose(-2, -1))/torch.tensor(k.size(-1)**0.5)
        attention_sim = F.softmax(scaled_sims)
        attention_scores = attention_sim@v

        return attention_scores


class MaskedSelfAttention(nn.Module):
    def __init__(self, d_model, row_dim, col_dim):
        super(MaskedSelfAttention, self).__init__()
        self.d_model = d_model
        self.row_dim = row_dim
        self.col_dim = col_dim
        self.W_q = nn.Linear(self.d_model, self.d_model)
        self.W_k = nn.Linear(self.d_model, self.d_model)
        self.W_v = nn.Linear(self.d_model, self.d_model)

    def forward(self, x, mask=None):
        q = self.W_q(x)
        k = self.W_k(x)
        v = self.W_v(x)

        scaled_sims = q@(k.transpose(-2, -1))/torch.tensor(k.size(self.col_dim)**0.5)

        if mask is not None:
            scaled_sims = scaled_sims.masked_fill(mask=mask, value=1e-20)
        
        attention_sim = F.softmax(scaled_sims)
        attention_scores = attention_sim@v

        return attention_scores


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.d_model = d_model
        self.num_heads = num_heads

        assert d_model % num_heads == 0
        
        self.d_k = d_model // num_heads
        
        self.qkv_proj = nn.Linear(self.d_model, self.d_model*3)

        self.W_o = nn.Linear(self.d_model, self.d_model)

    def forward(self, x, mask=None):
        qkv = self.qkv_proj(x)
        qkv = rearrange(
            qkv, "b s (h d) -> b h s d", h=self.num_heads, d=3*self.d_k
        )
        q, k, v = qkv.chunk(3, dim=-1)

        scaled_sims = q@(k.transpose(-2, -1))/torch.tensor(q.size(-1)**0.5)

        if mask is not None:
            scaled_sims = scaled_sims.masked_fill(mask=mask, value=1e-20)

        attention_scores = F.softmax(scaled_sims)@v
        attention_scores = rearrange(attention_scores, "b h s d -> b s (h d)")
        
        attention_out = self.W_o(attention_scores)

        return attention_out
