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


class BlockUpsampleAdd(nn.Module):

    def __init__(self, **kwargs):
        super(BlockUpsampleAdd, self).__init__()

        list__in_channels = kwargs["list__in_channels"]
        out_channels = kwargs["out_channels"]

        in_channels_1 = list__in_channels[0]
        in_channels_2 = list__in_channels[1]

        self.conv1 = nn.Conv2d(
            in_channels=in_channels_1,
            out_channels=out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(out_channels)

        self.conv2 = nn.Conv2d(
            in_channels=in_channels_2,
            out_channels=out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x, y):
        _, _, H, W = y.size()
        x = F.interpolate(x, size=(H, W), mode="nearest")
        x = self.conv1(x)
        x = self.bn1(x)

        y = self.conv2(y)
        y = self.bn2(y)

        out = x + y

        return out


class HeadDetection(nn.Module):

    def __init__(self, **kwargs):
        super(HeadDetection, self).__init__()

        self.num__anchors = kwargs["num__anchors"]
        self.num__classes = kwargs["num__classes"]

        out_channels = self.num__anchors * (5 + self.num__classes)

        self.list__block = nn.ModuleList(
            [
                nn.Conv2d(
                    in_channels=256,
                    out_channels=out_channels,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                    bias=True,
                )
                for _ in range(3)
            ]
        )

    def forward(self, list__x):
        list__x = [block(x) for block, x in zip(self.list__block, list__x)]
        list__x = [
            x.view(
                x.size(0),
                self.num__anchors,
                5 + self.num__classes,
                x.size(2),
                x.size(3),
            )
            .permute(0, 1, 3, 4, 2)
            .contiguous()
            for x in list__x
        ]
        return list__x
