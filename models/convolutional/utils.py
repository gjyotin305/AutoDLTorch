import torch.nn as nn

class ConvBlock(nn.Module):
    def __init__(
        self,
        activation_fn: nn.Module,
        in_channels: int,
        out_channels: int,
        kernel_size: int, 
        padding: int = 1,
        batch_norm: bool = False,
        stride: int = 1,
        pooling: nn.Module = None
    ):
        super().__init__(ConvBlock, self)
        self.activation_fn = activation_fn
        self.pooling = pooling
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            stride=stride,
            padding=padding,
            kernel_size=kernel_size
        )
        if batch_norm:
            self.batch_norm = nn.BatchNorm2d()
        else:
            self.batch_norm = None
    
    def forward(self, x):
        out = self.conv(x)
        if self.batch_norm is not None:
            out = self.batch_norm(out)
        out = self.activation_fn(out)
        if isinstance(self.pooling, nn.Module):
            out = self.pooling(out)
        return out