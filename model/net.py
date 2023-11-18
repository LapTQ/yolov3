import os
import sys
from pathlib import Path
import yaml

import torch
import torch.nn as nn


FILE = Path(__file__).resolve()
ROOT = FILE.parent.parent

if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from utils.general import check_file, select_device, load_yaml




class YOLOv3(nn.Module):

    def __init__(self, config, in_channels):
        super().__init__()

        anchors = config['anchors']
        nc = config['nc']
        backbone = config['backbone']

        out_channels = (len(anchors[0]) // 2) * (nc + 5)

        self.layers = []



    def forward(self, x):
        return self.model(x)



if __name__ == '__main__':
    cfg = check_file(ROOT / 'config/yolov3.yaml', mode='yaml')
    cfg = load_yaml(cfg)
    print(cfg)
    device = select_device('cpu')


    im = torch.rand(2, 3, 640, 640).to(device)


