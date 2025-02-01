import torch
import torch.nn as nn
from ._convolution import ConvBlock

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


class ConvAutoEncoder(nn.Module):
    def __init__(
        self, 
        activation_fn: nn.Module,
        conv_activation_fn: nn.Module,
        in_channels: int,
        hidden_channels: int,
        kernel_size: int,
        padding: int,
        n_blocks_enc: int,
    ):
        super().__init__(ConvAutoEncoder, self)
        self.conv_activation_fn = conv_activation_fn
        self.activation_fn = activation_fn
        self.enc_blocks = n_blocks_enc

        self.input_enc = [
            ConvBlock(
                in_channels=in_channels,
                out_channels=hidden_channels,
                kernel_size=kernel_size,
                padding=padding
            )
        ]

        self.enc_list = [
            ConvBlock(
                in_channels=hidden_channels*(x+1), 
                out_channels=hidden_channels*(x+2), 
                kernel_size=kernel_size, 
                padding=padding
            ) for x in range(self.enc_blocks)
        ]

        self.input_enc.extend(self.enc_list)

        self.conv_enc_block = nn.Sequential(*self.input_enc)

        self.encoder = nn.Sequential(

        )
        self.decoder = nn.Sequential(

        )

nn.ReLU()