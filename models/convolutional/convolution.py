import torch.nn as nn
from einops import rearrange
from .utils import ConvBlock
from ..base.simple import LazyANN

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
        n_layers_conv: int,
        hidden_dim: int,
        pool_kernel,
        n_classes: int,
        act_fn: nn.Module = nn.ReLU(),
    ):
        super(CNN2DClassification, self).__init__()
        self.in_channels = in_channels
        self.hidden_dim = hidden_dim
        self.growth_rate = growth_rate
        self.padding = padding
        self.kernel_size = kernel_size
        self.act_fn = act_fn
        self.n_layers_conv = n_layers_conv
        self.n_classes = n_classes
        self.pool_kernel = pool_kernel
        self._make_conv_block()
        self._make_fc_block()

    def _make_conv_block(self):
        """
        AutoCreate Ideal Combination
        """
        self.conv_list = [
            ConvBlock(
                type_in="2d",
                in_channels=self.in_channels,
                out_channels=self.growth_rate,
                kernel_size=self.kernel_size,
                stride=1,
                padding=1,
                act_fn=nn.ReLU()
            )
        ]
        self.mid_conv = [
            nn.Sequential(
                ConvBlock(
                    type_in="2d",
                    in_channels=self.growth_rate*i,
                    out_channels=self.growth_rate*i,
                    kernel_size=self.kernel_size,
                    stride=1,
                    padding=1,
                    act_fn=nn.ReLU(),
                    pool_=self.pool_kernel
                ),
                ConvBlock(
                    type_in="2d",
                    in_channels=self.growth_rate*i,
                    out_channels=self.growth_rate*(i+1),
                    kernel_size=self.kernel_size,
                    stride=1,
                    padding=1,
                    act_fn=nn.ReLU()
                )
            )
            for i in range(1, self.n_layers_conv)
        ]
        self.conv_list.extend(self.mid_conv)
        self.conv = nn.Sequential(*self.conv_list)

    def _make_fc_block(self):
        self.fc = LazyANN(
            hidden_dim=self.hidden_dim,
            out_dim=self.n_classes,
            act_fn=nn.ReLU()
        )

    def forward(self, x):
        """
        Rethink of a better way.
        """
        out = self.conv(x)
        out = rearrange(out, "b c h w -> b (c h w)")
        out = self.fc(out)
        return out


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