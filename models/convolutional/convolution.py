import torch.nn as nn
from .utils import ConvBlock

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


class TinyResNet(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        pass


class TinyUNet(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        pass


class TinyInceptionNet(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        pass


class MobileNet(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        pass


class TinyDenseNet(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        pass


class ConvNext(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        pass