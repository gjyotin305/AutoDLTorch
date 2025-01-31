import torch
import torch.nn as nn


class SimpleAutoEncoder(nn.Module):
    def __init__(
            self, 
            input_dim, 
            hidden_dim, 
            output_dim, 
            n_layer
        ):
        super().__init__(SimpleAutoEncoder, self)
        self.in_model = nn.Linear(input_dim, hidden_dim)
        self.interim_model = nn.Sequential(*[
            nn.Linear(hidden_dim, hidden_dim) for _ in range(n_layer)
        ])
        self.out_model = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        out = self.in_model(x)
        out = self.interim_model(out)
        out = self.out_model(out)
        return out


class AutoEncoder(nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_dim,
        activation_fn,
        output_dim,
        n_layers
    ):
        super().__init__(AutoEncoder, self)
        self.in_model = nn.Linear(input_dim, hidden_dim)
        self.act_fn = activation_fn
        self.interim_model = nn.Sequential(*[
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                self.act_fn
            ) for _ in range(n_layers)]
        )
        self.out_model = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        out = self.in_model(x)
        out = self.interim_model(out)
        out = self.out_model(x)
        return out

