import torch
from einops import rearrange
import torch.nn as nn
import torch.nn.functional as F


class BottleNeck(nn.Module):
    def __init__(self, in_channels, kernel_size, padding, pool):
        super(BottleNeck, self).__init__()
        self.in_channels = in_channels
        self.kernel_size = kernel_size
        self.padding = padding
        self.pool = pool
        self.model = nn.Sequential(
            nn.Conv2d(
                in_channels=self.in_channels,
                out_channels=self.in_channels,
                kernel_size=self.kernel_size,
                padding=padding
            ),
            nn.BatchNorm2d(self.in_channels),
            nn.ReLU(True),
            nn.MaxPool2d(self.pool)
        )
    
    def forward(self, x):
        out = self.model(x)
        return out

class BasicBlock(nn.Module):
    def __init__(self, in_channel, kernel_size, padding, out_channel=None):
        super(BasicBlock, self).__init__()
        self.in_channel = in_channel
        self.kernel_size = kernel_size
        self.padding = padding
        self.out_channel = out_channel
        if self.out_channel is None:
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
        else:
            self.model = nn.Sequential(
                nn.Conv2d(
                    in_channels=self.in_channel,
                    out_channels=self.out_channel,
                    kernel_size=self.kernel_size,
                    padding=self.padding
                ),
                nn.BatchNorm2d(self.out_channel),
                nn.ReLU(True),
                nn.Conv2d(
                    in_channels=self.out_channel,
                    out_channels=self.out_channel,
                    kernel_size=self.kernel_size,
                    padding=self.padding
                ),
                nn.BatchNorm2d(self.out_channel)
            )

    
    def forward(self, x):
        identity = x
        out = self.model(x)
        
        if self.out_channel is None:
            out += identity

        out = F.relu(out)
        return out

class ResNetArch(nn.Module):
    def __init__(self, num_blocks, classifier_depth, num_classes):
        super(ResNetArch, self).__init__()
        self.num_blocks = num_blocks
        self.classifier_depth = classifier_depth
        self.num_classes = num_classes
        self.embedding_size = 512
        self._make_architecture()
    
    def _make_architecture(self):
        self.conv_list = [
            BasicBlock(in_channel=3, kernel_size=3, padding=1, out_channel=16)
        ]
        self.conv_list.extend(
            [
                BasicBlock(in_channel=16, kernel_size=3, padding=1),
                BasicBlock(
                    in_channel=16, kernel_size=3, padding=1, out_channel=64),
                BottleNeck(
                    in_channels=64, padding=1, kernel_size=3, pool=(2, 2)),
                BasicBlock(in_channel=64, kernel_size=3, padding=1, out_channel=128),
                BasicBlock(in_channel=128, kernel_size=3, padding=1),
                BasicBlock(in_channel=128, kernel_size=3, padding=1, out_channel=256),
                BottleNeck(
                    in_channels=256, kernel_size=3, padding=1, pool=(2,2)
                )
            ]
        )
        self.conv_block = nn.Sequential(*self.conv_list)
        self.classifier = nn.Sequential(
            nn.LazyLinear(self.embedding_size),
            nn.Linear(self.embedding_size, self.embedding_size),
            nn.Linear(self.embedding_size, self.num_classes)
        )
    
    def forward(self, x):
        out = self.conv_block(x)
        out = rearrange(out, "b c h w -> b (c h w)")
        out = self.classifier(out)
        return out