from typing import List
import torch.nn as nn

class SimpleAutoEncoder(nn.Module):
    def __init__(
            self, 
            input_dim, 
            hidden_dim,
            n_layers
        ):
        super(SimpleAutoEncoder, self).__init__()
        self.in_model = nn.Linear(input_dim, hidden_dim)
        self.encoder = nn.Sequential(*[
            nn.LazyLinear(int((1/n)*hidden_dim)) for n in range(1, n_layers)
        ])
        self.decoder = nn.Sequential(*[
            nn.LazyLinear(int((1/n)*hidden_dim)) for n in range(n_layers-1, 0, -1)
        ])
        self.out_model = nn.LazyLinear(input_dim)

    def forward(self, x):
        out = self.in_model(x)
        out = self.encoder(out)
        out = self.decoder(out)
        out = self.out_model(out)
        return out


class AutoEncoder(nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_dim,
        activation_fn: nn.Module,
        n_layers
    ):
        super(AutoEncoder, self).__init__()
        self.in_model = nn.Linear(input_dim, hidden_dim)
        self.act_fn = activation_fn
        self.encoder = nn.Sequential(*[
            nn.Sequential(
                nn.LazyLinear(int((1/n)*hidden_dim)),
                self.act_fn
            ) for n in range(1, n_layers)
        ])
        self.decoder = nn.Sequential(*[
            nn.Sequential(
                nn.LazyLinear(int((1/n)*hidden_dim)),
                self.act_fn
            ) for n in range(n_layers-1, 0, -1)
        ])
        self.out_model = nn.LazyLinear(input_dim)

    def forward(self, x):
        out = self.in_model(x)
        out = self.encoder(out)
        out = self.decoder(out)
        out = self.out_model(out)
        return out

    def encoder_forward(self, x):
        out = self.in_model(x)
        out = self.encoder(x)
        return out

class ConvAutoEncoder(nn.Module):
    def __init__(
        self,
        input_encoder_: List[nn.Module],
        out_decoder_: List[nn.Module]
    ):
        super(ConvAutoEncoder, self).__init__()
        self.encoder = nn.Sequential(*input_encoder_)
        self.decoder = nn.Sequential(*out_decoder_)
    
    def forward(self, x):
        out = self.encoder(x)
        out = self.decoder(out)
        return out