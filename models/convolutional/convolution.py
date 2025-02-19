import torch.nn as nn
from .utils import ConvBlock

class CNN2DClassification(nn.Module):
    def __init__(
        self,
        in_channels: int,
        growth_rate: int,
        kernel_size: int,
        padding: int,
        n_layers: int,
        pool_kernel,
        n_classes: int,
        act_fn: nn.Module = nn.ReLU(),
    ):
        super(CNN2DClassification, self).__init__()
        self.in_channels = in_channels
        self.growth_rate = growth_rate
        self.padding = padding
        self.kernel_size = kernel_size
        self.act_fn = act_fn
        self.n_layers = n_layers
        self.n_classes = n_classes
        self.pool_kernel = pool_kernel


    def _make_conv_block(self):
        """
        AutoCreate Ideal Combination
        """
        raise NotImplementedError
        


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