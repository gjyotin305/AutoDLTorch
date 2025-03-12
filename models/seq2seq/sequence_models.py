import torch
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


class GatedArch(nn.Module):
    def __init__(self, input_dim, recurrent_dim, output_dim):
        super(GatedArch, self).__init__()
        self.input_dim = input_dim
        self.recurrent_dim = recurrent_dim
        self.output_dim = output_dim
        self._init_layers()
        
    def _init_layers(self):
        self.x2h = nn.Linear(self.input_dim, self.recurrent_dim)
        self.h2h = nn.Linear(self.recurrent_dim, self.recurrent_dim, bias=False)
        self.x2z = nn.Linear(self.input_dim, self.recurrent_dim)
        self.h2z = nn.Linear(self.recurrent_dim, self.recurrent_dim, bias=False)
        self.h2y = nn.Linear(self.recurrent_dim, self.output_dim)
    
    def forward(self, x):
        """    
        Process:

        - Initial Recurrent Update:
        `Ht_candidate = ReLU(Wxh @ Xt + Whh @ Ht_1 + bh)`
        - Full Update:
        The same equation as the initial recurrent update (for clarity).
        - Forget Gate:
        `Zt = ActFn(Wxz @ Xt + Whz @ Ht_1 + bz)`
        - Gated Recurrent Update:
        `Ht = Zt * Ht_1 + (1 - Zt) * Ht_candidate`
        """

        h0 = x.new_zeros(x.size(0), self.h2y.weight.size(1))

        for t in range(x.size(1)):
            z = torch.sigmoid(self.x2z(x[:, t] + self.h2z(h0)))
            hb = F.relu(self.x2h(x[:, t]) + self.h2h(h0))
            h0 = z*h0 + (1-z)*hb
        
        return self.h2y(h0)