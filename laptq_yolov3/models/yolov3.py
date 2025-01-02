import torch
from torch import nn
import torch.nn.functional as F

from laptq_yolov3.models import Darknet53, FPN, HeadDetection


class YOLOv3(nn.Module):

    def __init__(self, **kwargs):
        super(YOLOv3, self).__init__()

        self.num__anchors = kwargs["num__anchors"]
        self.num__classes = kwargs["num__classes"]

        self.backbone = Darknet53()
        self.neck = FPN()
        self.head1 = HeadDetection(
            out_channels=self.num__anchors * (5 + self.num__classes),
        )
        self.head2 = HeadDetection(
            out_channels=self.num__anchors * (5 + self.num__classes),
        )
        self.head3 = HeadDetection(
            out_channels=self.num__anchors * (5 + self.num__classes),
        )

    
    def forward(self, x):
        x1, x2, x3 = self.backbone(x)
        x1, x2, x3 = self.neck([x1, x2, x3])
        x1 = self.head1(x1)
        x2 = self.head2(x2)
        x3 = self.head3(x3)

        x1 = x1.view(x1.size(0), self.num__anchors, 5 + self.num__classes, x1.size(2), x1.size(3)).permute(0, 1, 3, 4, 2).contiguous()
        x2 = x2.view(x2.size(0), self.num__anchors, 5 + self.num__classes, x2.size(2), x2.size(3)).permute(0, 1, 3, 4, 2).contiguous()
        x3 = x3.view(x3.size(0), self.num__anchors, 5 + self.num__classes, x3.size(2), x3.size(3)).permute(0, 1, 3, 4, 2).contiguous()

        return [x1, x2, x3]


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