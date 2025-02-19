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
        super(CNN, self).__init__()
        self.in_channels = in_channels
        self.conv_block = nn.Conv2d(
            in_channels=in_channels,
            out_channels=
        )

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