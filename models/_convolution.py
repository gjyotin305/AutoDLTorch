import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    def __init__(
        self,
        activation_fn: nn.Module,
        pooling: str,
        in_channels: int, 
        out_channels: int,
        kernel_size: int, 
        padding: int,
    ):
        super().__init__(ConvBlock, self)
        self.activation_fn = activation_fn
        self.pooling = pooling
        if isinstance(self.pooling, nn.Module):
            self.block = nn.Sequential(
                nn.Conv2d(
                    in_channels=in_channels, 
                    out_channels=out_channels, 
                    kernel_size=kernel_size,
                    padding=padding
                ),
                nn.BatchNorm2d(out_channels),
                self.activation_fn,
                self.pooling
            )
        else:
            self.block = nn.Sequential(
                nn.Conv2d(
                    in_channels=in_channels, 
                    out_channels=out_channels, 
                    kernel_size=kernel_size,
                    padding=padding
                ),
                nn.BatchNorm2d(out_channels),
                self.activation_fn,
            )
    
    
    def forward(self, x):
        out = self.block(x)
        return out