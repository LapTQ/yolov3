from collections import OrderedDict

import torch
from torch import nn
import torch.nn.functional as F

from laptq_yolov3.models import BlockConv, BlockBottleNeck


class Darknet53(nn.Module):

    def __init__(self, **kwargs):
        super(Darknet53, self).__init__()

        list__info__block = [
            (1, BlockConv, [3, 32, 3, 1, 1], False),
            (1, BlockConv, [32, 64, 3, 1, 2], False),
            (1, BlockBottleNeck, [64, 64, None, None, None], False),
            (1, BlockConv, [64, 128, 3, 1, 2], False),
            (2, BlockBottleNeck, [128, 128, None, None, None], False), # C2 /4
            (1, BlockConv, [128, 256, 3, 1, 2], False),
            (8, BlockBottleNeck, [256, 256, None, None, None], True), # C3 /8
            (1, BlockConv, [256, 512, 3, 1, 2], False),
            (8, BlockBottleNeck, [512, 512, None, None, None], True),   # C4 /16
            (1, BlockConv, [512, 1024, 3, 1, 2], False),
            (4, BlockBottleNeck, [1024, 1024, None, None, None], True), # C5 /32
        ]

        self.list__block = nn.ModuleList()
        self.list__to_capture = []
        for num__repeat, cls__block, list__param, to_capture in list__info__block:
            for i_repeat in range(num__repeat):
                in_channels = list__param[0]
                out_channels = list__param[1]
                kernel_size = list__param[2]
                padding = list__param[3]
                stride = list__param[4]

                block = cls__block(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    padding=padding,
                    stride=stride,
                )

                self.list__block.append(block)
                self.list__to_capture.append(i_repeat == num__repeat - 1 and to_capture)

    def forward(self, x):
        list__captured = []
        for block, to_capture in zip(self.list__block, self.list__to_capture):
            x = block(x)
            if to_capture:
                list__captured.append(x)
        return list__captured


if __name__ == "__main__":

    device = torch.device("cuda:0")
    model = Darknet53().to(device)

    print(model)

    x = torch.randn((4, 3, 416, 416)).to(device)
    out = model(x)

    print([_.shape for _ in out])
