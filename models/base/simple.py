import torch.nn as nn


class LazyANN(nn.Module):
    def __init__(
        self, 
        hidden_dim, 
        out_dim,
        act_fn
    ):
        super(LazyANN, self).__init__()
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.act_fn = act_fn
        self._make_block()

    def _make_block(self):
        self.model = nn.Sequential(
            nn.LazyLinear(self.hidden_dim),
            self.act_fn,
            nn.LazyLinear(self.hidden_dim),
            self.act_fn,
            nn.LazyLinear(self.out_dim)
        )
        print("Model Constructed")
    
    def forward(self, x):
        out = self.model(x)
        return out

class ANN(nn.Module):
    def __init__(
        self, 
        in_dim, 
        hidden_dim, 
        out_dim, 
        act_fn
    ):
        super(ANN, self).__init__()
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
        super(DNN, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.hid_dim = hid_dim
        self.act_fn = act_fn
        self.n_layers = n_layers
        self.model = ANN(
            in_dim=self.in_dim,
            hidden_dim=self.hid_dim,
            out_dim=self.out_dim,
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
        