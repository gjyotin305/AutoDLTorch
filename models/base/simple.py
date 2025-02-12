import torch.nn as nn

class ANN(nn.Module):
    def __init__(
        self, 
        in_dim, 
        hidden_dim, 
        out_dim, 
        act_fn
    ):
        super().__init__(ANN, self)
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.hid_dim = hidden_dim
        self.act_fn = act_fn
        self.in_layer = nn.Linear(self.in_dim, self.hid_dim)
        self.mid_layer = nn.Sequential(
            nn.Linear(self.hid_dim, self.hid_dim),
            self.act_fn
        )
        self.out_layer = nn.Linear(self.hid_dim, self.out_dim)
    
    def forward(self, x):
        out = self.in_layer(x)
        out = self.mid_layer(out)
        out = self.out_layer(out)
        return out

class DNN(nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim, n_layers, act_fn):
        super().__init__(DNN, self)
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.hid_dim = hid_dim
        self.act_fn = act_fn
        self.n_layers = n_layers
        self.model = ANN(
            in_dim=self.in_dim,
            hidden_dim=self.hid_dim,
            out_dim=self.hid_dim,
            act_fn=self.act_fn
        )
        self.model.mid_layer = nn.Sequential(
            *[
                nn.Sequential(
                    nn.Linear(self.hid_dim, self.hid_dim),
                    self.act_fn
                ) for _ in range(self.n_layers)
            ]
        )
    
    def forward(self, x):
        out = self.model(x)
        return out
        