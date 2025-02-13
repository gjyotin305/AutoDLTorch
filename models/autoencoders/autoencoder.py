from typing import List
import torch.nn as nn
from ..convolutional.convolution import ConvBlock


class ConvDecoderBlock(nn.Module):
    def __init__(
        self, 
        in_dim, 
        growth_rate,
        kernel_size,
        padding,
        activation_fn
    ):
        super(ConvDecoderBlock, self).__init__()
        self.act_fn = activation_fn
        self.decoder_block = nn.Sequential(
            nn.Upsample(
                scale_factor=growth_rate, 
                mode='bilinear', 
                align_corners=True
            ),
            nn.Conv2d(
                in_dim, 
                int((1/growth_rate)*in_dim), 
                kernel_size=kernel_size, 
                padding=padding
            ),
            nn.BatchNorm2d(growth_rate*in_dim),
            self.act_fn
        )

    def forward(self, x):
        out = self.encoder(x)
        return out

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
        in_channels,
        final_activation,
        growth_rate,
        n_layers,
        kernel_size,
        padding
    ):
        super(ConvAutoEncoder, self).__init__()
        
        self.in_encoder = ConvBlock(
            activation_fn=nn.ReLU(),
            in_channels=in_channels,
            out_channels=growth_rate,
            kernel_size=kernel_size,
            padding=padding
        )
        self.encoder = nn.Sequential(
            *[
                nn.Sequential(
                    ConvBlock(
                        activation_fn=nn.ReLU(),
                        in_channels=growth_rate*n,
                        out_channels=growth_rate*n,
                        kernel_size=kernel_size,
                        padding=padding
                    ),
                    ConvBlock(
                        activation_fn=nn.ReLU(),
                        in_channels=growth_rate*n,
                        out_channels=growth_rate*(n+1),
                        kernel_size=kernel_size,
                        padding=padding,
                        pooling=nn.MaxPool2d((growth_rate, growth_rate))
                    )
                ) for n in range(1, n_layers)
            ]
        )
    
    def forward(self, x):
        out = self.encoder(x)
        out = self.decoder(out)
        return out