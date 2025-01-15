import torch
from torch import nn
import torch.nn.functional as F

from laptq_yolov3.models import Darknet53, FPN, HeadDetection


class YOLOv3(nn.Module):

    def __init__(self, **kwargs):
        super(YOLOv3, self).__init__()

        self.num__anchors = kwargs["num__anchors"]
        self.num__classes = kwargs["num__classes"]
        self.list__anchor_box__wh = kwargs["list__anchor_box__wh"]
        self.list__anchor_box__i_layer = kwargs["list__anchor_box__i_layer"]

        self.backbone = Darknet53()
        self.neck = FPN()
        self.head = HeadDetection(
            num__anchors=self.num__anchors,
            num__classes=self.num__classes,
        )

    def forward(self, x):
        x = self.backbone(x)
        x = self.neck(x)
        x = self.head(x)
        return x


if __name__ == "__main__":

    device = torch.device("cuda:0")
    model = YOLOv3(
        num__anchors=3,
        num__classes=90,
    ).to(device)

    print(model)

    x = torch.randn((4, 3, 416, 416)).to(device)
    out = model(x)

    print([_.shape for _ in out])
