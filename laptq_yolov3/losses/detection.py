import torch


class LossYOLOv3:

    def __init__(self, **kwargs):
        self.model = kwargs["model"]

        self.list__anchor_box__wh = self.model.list__anchor_box__wh
        self.list__anchor_box__i_layer = self.model.list__anchor_box__i_layer

    def __call__(self, batch__output, batch__lbl):
        """
        batch__output: num-layers x [Size([batch-size, num-anchors, H, W, len([tx, ty, tw, th, objness, *class-score])])]
        batch__lbl: Size([sum--of--num-boxes, len([id__img, xcn, ycn, wn, hn, id_class])])
        """
        list__layer_size = [_.shape[2] for _ in batch__output]
        

        for i__layer, layer_size in enumerate(list__layer_size):
            batch__lbl__xcyc = batch__lbl[:, 1:3] * layer_size
            batch__lbl__cxcy = (batch__lbl__xcyc // 1).int()  # cell offset
            batch__lbl__txty = -torch.log(1 / (batch__lbl__xcyc - batch__lbl__cxcy + 1e-9) - 1)
            print(batch__lbl__cxcy)
            print(batch__lbl__txty)
            exit()
