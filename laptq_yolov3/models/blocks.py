import torch
from torch import nn
import torch.nn.functional as F


class BlockConv(nn.Module):

    def __init__(self, **kwargs):
        super(BlockConv, self).__init__()

        in_channels = kwargs["in_channels"]
        out_channels = kwargs["out_channels"]
        kernel_size = kwargs["kernel_size"]
        padding = kwargs["padding"]
        stride = kwargs["stride"]

        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=False,
        )
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = F.relu(x)
        return x


class BlockBottleNeck(nn.Module):

    def __init__(self, **kwargs):
        super(BlockBottleNeck, self).__init__()

        in_channels = kwargs["in_channels"]
        out_channels = kwargs["out_channels"]

        self.block_conv_1 = BlockConv(
            in_channels=in_channels,
            out_channels=in_channels // 2,
            kernel_size=1,
            padding=0,
            stride=1,
        )
        self.block_conv_2 = BlockConv(
            in_channels=in_channels // 2,
            out_channels=out_channels,
            kernel_size=3,
            padding=1,
            stride=1,
        )
        self.conv3 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False,
        )

    def forward(self, x):
        identity = x
        x = self.block_conv_1(x)
        x = self.block_conv_2(x)
        identity = self.conv3(identity)
        x += identity
        x = F.relu(x)
        return x
