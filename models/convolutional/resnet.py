import torch
import torch.nn as nn
import torch.nn.functional as F

class BasicBlock(nn.Module):
    def __init__(self, in_channel,kernel_size, padding):
        super(BasicBlock, self).__init__()
        self.in_channel = in_channel
        self.kernel_size = kernel_size
        self.padding = padding
        self.model = nn.Sequential(
            nn.Conv2d(
                in_channels=self.in_channel,
                out_channels=self.in_channel,
                kernel_size=self.kernel_size,
                padding=self.padding
            ),
            nn.BatchNorm2d(self.in_channel),
            nn.ReLU(True),
            nn.Conv2d(
                in_channels=self.in_channel,
                out_channels=self.in_channel,
                kernel_size=self.kernel_size,
                padding=self.padding
            ),
            nn.BatchNorm2d(self.in_channel)
        )
    
    def forward(self, x):
        identity = x
        out = self.model(x)
        out += identity
        out = F.relu(out)
        return out

class ResNetArch(nn.Module):
    def __init__(self, input_dim):
        super(ResNetArch, self).__init__()
        self.input_dim = input_dim
    
    def _make_architecture(self):
        raise NotImplementedError