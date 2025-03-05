import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class ElmanNetwork(nn.Module):
    def __init__(self, input_dim, recurrent_dim, output_dim):
        super().__init__()
        self.x2h = nn.Linear(input_dim, recurrent_dim)
        self.h2h = nn.Linear(recurrent_dim, recurrent_dim, bias=False)
        self.h2y = nn.Linear(recurrent_dim, output_dim)

    def forward(self, x):
        # ht = ReLU(WxhXt + WhhHt-1 + Bh)
        h = x.new_zeros(x.size(0), self.h2y.weight.size(1))
        for t in range(input.size(1)):
            h = F.relu(self.x2h(x[:, t]) + self.h2h(h))
        
        return self.h2y(h)

class RNNArch(nn.Module):
    def __init__(self):
        super().__init__()
        raise NotImplementedError


class LSTMArch(nn.Module):
    def __init__(self):
        super().__init__()
        raise NotImplementedError