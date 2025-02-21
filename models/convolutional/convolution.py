import torch.nn as nn
from .utils import ConvBlock

class CNN2DClassification(nn.Module):
    """
    Best CNN out of all the below ones.
    """
    def __init__(
        self,
        in_channels: int,
        growth_rate: int,
        kernel_size: int,
        padding: int,
        n_layers_fc: int,
        n_layers_conv: int,
        hidden_dim: int,
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
        self.n_layers_conv = n_layers_conv
        self.n_layers_fc = n_layers_fc
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


class ResNet(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        pass


class DenseNet(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        pass


class UNet(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        pass


class InceptionNet(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        pass


class MobileNet(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        pass


class ConvNext(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        pass