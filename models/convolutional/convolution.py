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


class CNN(nn.Module):
    def __init__(
        self, 
        in_channels: int,
        growth_rate: int,
        hid_dim: int, 
        out_dim: int,
        kernel_size: int,
        padding: int,
        n_layers: int,
        n_classes: int,
        pooling: nn.Module,
        act_fn: nn.Module = nn.ReLU(),
    ):
        super().__init__(CNN, self)
        self.in_channels = in_channels
        self.n_layers = n_layers
        self.out_dim = out_dim
        self.hid_dim = hid_dim
        self.k_size = kernel_size
        self.padding = padding
        self.start_conv = ConvBlock(
            in_channels=in_channels,
            out_channels=growth_rate,
            kernel_size=kernel_size,
            padding=padding,
            activation_fn=act_fn,
            pooling=pooling,
            batch_norm=True
        )
        self.conv_blocks = nn.Sequential(
            *[
                ConvBlock(
                    in_channels=growth_rate*(n-1),
                    out_channels=growth_rate*n,
                    kernel_size=kernel_size,
                    padding=padding,
                    activation_fn=act_fn,
                    pooling=pooling,
                    batch_norm=True
                )
                for n in range(2, n_layers-1)
            ]
        )
        self.fc = nn.LazyLinear(n_classes)

    def forward(self, x):
        """
        Rethink of a better way.
        """
        return NotImplementedError
        # out = self.start_conv(x)
        # out = self.conv_blocks(out)
        # out = self.fc(out)
        # return out