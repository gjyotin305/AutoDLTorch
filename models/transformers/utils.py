import torch 
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len, device="cpu"):
        super(PositionalEncoding, self).__init__()
        self.encoding = torch.zeros(max_len, d_model, device=device)
        self.encoding.requires_grad = False

        pos = torch.arange(0, max_len, device=device)
        pos = rearrange(pos, "(h d) -> h d", d=1)

        _2i = torch.arange(0, d_model, step=2, device=device).float()

        self.encoding[:, 0::2] = torch.sin(pos/(10000 ** (_2i/d_model)))
        self.encoding[:, 1::2] = torch.sin(pos/(10000 ** (_2i/d_model)))

    def forward(self, x):
        
        batch_size, seq_len, _ = x.size()

        return x + self.encoding[:seq_len, :]