import torch
import torch.nn as nn
import torch.nn.functional as F

class SelfAttention(nn.Module):
    def __init__(self, d_model, row_dim, col_dim):
        super(SelfAttention, self).__init__()
        self.d_model = d_model
        self.row_dim = row_dim
        self.col_dim = col_dim

        self.W_q = nn.Linear(self.d_model, self.d_model)
        self.W_k = nn.Linear(self.d_model, self.d_model)
        self.W_v = nn.Linear(self.d_model, self.d_model)
    
    def forward(self, x):
        q = self.W_q(x)
        k = self.W_k(x)
        v = self.W_v(x)

        scaled_sims = q@k.T/torch.tensor(k.size(self.col_dim)**0.5)
        attention_sim = F.softmax(scaled_sims)
        attention_scores = attention_sim@v

        return attention_scores


class MaskedSelfAttention(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        raise NotImplementedError


class MultiHeadAttention(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        raise NotImplementedError