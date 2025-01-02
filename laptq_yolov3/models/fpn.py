import torch
from torch import nn
import torch.nn.functional as F

from laptq_yolov3.models import BlockUpsampleAdd


class FPN(nn.Module):

    def __init__(self, **kwargs):
        super(FPN, self).__init__()

        self.upsampe_add_1 = BlockUpsampleAdd(
            list__in_channels=[1024, 512],
            out_channels=512,
        )

        self.upsampe_add_2 = BlockUpsampleAdd(
            list__in_channels=[512, 256],
            out_channels=256,
        )

        self.conv1 = nn.Conv2d(
            in_channels=1024,
            out_channels=256,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(256)

        self.conv2 = nn.Conv2d(
            in_channels=512,
            out_channels=256,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(256)

        self.conv3 = nn.Conv2d(
            in_channels=256,
            out_channels=256,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.bn3 = nn.BatchNorm2d(256)

    
    def forward(self, list__x):
        c3, c4, c5 = list__x

        p5 = c5
        p4 = self.upsampe_add_1(p5, c4)
        p3 = self.upsampe_add_2(p4, c3)

        p5 = self.conv1(p5)
        p5 = self.bn1(p5)

        p4 = self.conv2(p4)
        p4 = self.bn2(p4)

        p3 = self.conv3(p3)
        p3 = self.bn3(p3)

        return [p3, p4, p5]
    

