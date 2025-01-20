from scipy.optimize import linear_sum_assignment

    
from laptq_pyutils.ops import xcycwh__to__x1y1x2y2

import torch
from torchvision.ops import box_iou
from torch.nn.functional import one_hot
from torch.nn import BCEWithLogitsLoss

from laptq_yolov3.models import YOLOv3


class LossYOLOv3:

    def __init__(self, **kwargs):
        self.model: YOLOv3 = kwargs["model"]

        self.list__anchor_box__wh = torch.tensor(self.model.list__anchor_box__wh)
        self.list__anchor_box__i_layer = torch.tensor(
            self.model.list__anchor_box__i_layer
        )
        self.list__i_layer__scale = self.model.list__i_layer__scale

    def __call__(
        self,
        batch__output: torch.Tensor,
        batch__lbl: torch.Tensor,
    ):
        """
        batch__output: num-layers x [Size([batch-size, num-anchors, H, W, len([tx, ty, tw, th, objness, *class-score])])]
        batch__lbl: Size([sum--of--num-boxes, len([id__img, xcn, ycn, wn, hn, id_class])])
        """

        _ = self.assign__boxes(
            batch__output=batch__output,
            batch__lbl=batch__lbl,
        )
        batch__target__xc_yc = _["batch__target__xc_yc"]
        batch__target__w_h = _["batch__target__w_h"]
        batch__assigned__cx_cy = _["batch__assigned__cx_cy"]
        batch__assigned__tx_ty = _["batch__assigned__tx_ty"]
        batch__assigned__tw_th = _["batch__assigned__tw_th"]
        batch__assigned__iou = _["batch__assigned__iou"]
        batch__assigned__i_layer = _["batch__assigned__i_layer"]
        batch__assigned__i_abox = _["batch__assigned__i_abox"]

        batch_size = batch__output[0].shape[0]

        print('--------')
        print(batch__assigned__i_layer)
        print(batch__lbl[:, 0])

        loss__objness = 0.0
        loss__iou = 0.0
        loss__cls = 0.0

        # masks that are used to mark the anchor boxes on the model's output which are assigned to the ground truth boxes
        list__mask_output = []
        for i__layer, batch__output__layer_i in enumerate(batch__output):
            mask__output = torch.zeros_like(
                batch__output__layer_i[:, :, :, :, 0], dtype=torch.bool
            )  # Size([batch-size, num-anchors, H, W])

            for id__img in range(batch_size):
                mask__batch_lbl = (batch__lbl[:, 0] == id__img) & (
                    batch__assigned__i_layer == i__layer
                )
                mask__output[
                    id__img,
                    batch__assigned__i_abox[mask__batch_lbl],
                    batch__assigned__cx_cy[mask__batch_lbl, 1],
                    batch__assigned__cx_cy[mask__batch_lbl, 0],
                ] = True
                print('\t', list(zip(batch__assigned__i_abox[mask__batch_lbl].tolist(), batch__assigned__cx_cy[mask__batch_lbl, 1].tolist(), batch__assigned__cx_cy[mask__batch_lbl, 0].tolist())))
            print('--')
            list__mask_output.append(mask__output)

        print([_.sum() for _ in list__mask_output])

        for i__layer, (batch__output__layer_i, layer_scale, mask__output) in enumerate(
            zip(batch__output, self.list__i_layer__scale, list__mask_output)
        ):
            mask__batch_lbl__by__i_layer = batch__assigned__i_layer == i__layer
            print(i__layer, sum(mask__batch_lbl__by__i_layer))

            # output in layer i
            batch__output__layer_i__tx_ty = batch__output__layer_i[
                :, :, :, :, :2
            ]  # Size([batch-size, num-anchors, H, W, 2])
            batch__output__layer_i__tw_th = batch__output__layer_i[
                :, :, :, :, 2:4
            ]  # Size([batch-size, num-anchors, H, W, 2])
            batch__output__layer_i__objness = batch__output__layer_i[
                :, :, :, :, 4
            ]  # objectness Size([batch-size, num-anchors, H, W])
            batch__output__layer_i__class_score = batch__output__layer_i[
                :, :, :, :, 5:
            ]  # class score Size([batch-size, num-anchors, H, W, num-classes])

            # target in layer i
            batch__target__layer_i__xc_yc = batch__target__xc_yc[
                mask__batch_lbl__by__i_layer
            ]
            batch__target__layer_i__w_h = batch__target__w_h[
                mask__batch_lbl__by__i_layer
            ]
            batch__target__layer_i__xc_yc_w_h = torch.cat(
                [batch__target__layer_i__xc_yc, batch__target__layer_i__w_h], dim=1
            )

            batch__target__layer_i__objness = torch.zeros_like(
                batch__output__layer_i__objness
            )
            batch__target__layer_i__class_score = torch.zeros_like(
                batch__output__layer_i__class_score
            )
            for id__img in range(batch_size):
                mask__by__id_img = batch__lbl[:, 0] == id__img
                mask = mask__by__id_img & mask__batch_lbl__by__i_layer
                batch__target__layer_i__objness[
                    id__img,
                    batch__assigned__i_abox[mask],
                    batch__assigned__cx_cy[mask, 1],
                    batch__assigned__cx_cy[mask, 0],
                ] = batch__assigned__iou[mask]
                batch__target__layer_i__class_score[
                    id__img,
                    batch__assigned__i_abox[mask],
                    batch__assigned__cx_cy[mask, 1],
                    batch__assigned__cx_cy[mask, 0],
                ] = one_hot(
                    batch__lbl[mask, 5].long(), num_classes=self.model.num__classes
                ).float()

            # prior in layer i
            batch__prior__layer_i__cx_cy = batch__assigned__cx_cy[
                mask__batch_lbl__by__i_layer
            ]
            batch__prior__layer_i__pw_ph = self.list__anchor_box__wh[
                self.list__anchor_box__i_layer == i__layer
            ][batch__assigned__i_abox[mask__batch_lbl__by__i_layer]]

            # pred in layer i
            batch__pred__layer_i__sx_sy = torch.sigmoid(
                batch__output__layer_i__tx_ty[mask__output]
            )
            batch__pred__layer_i__xc_yc = (
                batch__prior__layer_i__cx_cy + batch__pred__layer_i__sx_sy
            )
            batch__pred__layer_i__w_h = (
                batch__prior__layer_i__pw_ph * layer_scale
            ) * torch.exp(batch__output__layer_i__tw_th[mask__output])
            batch__pred__layer_i__xc_yc_w_h = torch.cat(
                [batch__pred__layer_i__xc_yc, batch__pred__layer_i__w_h], dim=1
            )

            # iou
            batch__target__layer_i__x1_y1_x2_y2 = xcycwh__to__x1y1x2y2(
                batch__target__layer_i__xc_yc_w_h, backend="torch"
            )
            batch__pred__layer_i__x1_y1_x2_y2 = xcycwh__to__x1y1x2y2(
                batch__pred__layer_i__xc_yc_w_h, backend="torch"
            )
            mat__iou = box_iou(
                batch__target__layer_i__x1_y1_x2_y2, batch__pred__layer_i__x1_y1_x2_y2
            )

            # loss
            loss__objness += BCEWithLogitsLoss(reduction="mean")(
                batch__output__layer_i__objness,
                batch__target__layer_i__objness,
            )
            loss__iou += (1 - mat__iou).mean()
            loss__cls += BCEWithLogitsLoss(reduction="mean")(
                batch__output__layer_i__class_score[mask__output],
                batch__target__layer_i__class_score[mask__output],
            )

        loss = loss__objness + loss__iou + loss__cls

        return loss

    def assign__boxes(self, **kwargs):
        batch__output = kwargs["batch__output"]
        batch__lbl = kwargs["batch__lbl"]

        batch_size = batch__output[0].shape[0]

        list__layer_size = [
            _.shape[2] for _ in batch__output
        ]  # for input size 416x416, it should be [52, 26, 13]
        device = batch__lbl.device

        self.list__anchor_box__wh = self.list__anchor_box__wh.to(device)
        self.list__anchor_box__i_layer = self.list__anchor_box__i_layer.to(device)

        _batch__xc_yc = []
        _batch__w_h = []
        _batch__cx_cy = []
        _batch__tx_ty = []
        _batch__tw_th = []
        _batch__iou = []
        for i__layer, (layer_size, layer_scale) in enumerate(
            zip(list__layer_size, self.list__i_layer__scale)
        ):
            batch__target__layer_i__xc_yc = (
                batch__lbl[:, 1:3] * layer_size
            )  # Size([sum--of--num-boxes, 2])
            batch__assigned__layer_i__cx_cy = (
                batch__target__layer_i__xc_yc.int()
            )  # cell offset to top-left, Size([sum--of--num-boxes, 2])
            batch__target__layer_i__sx_sy = (
                batch__target__layer_i__xc_yc - batch__assigned__layer_i__cx_cy
            )  # relative offset to cell, Size([sum--of--num-boxes, 2])
            batch__assigned__layer_i__tx_ty = -torch.log(
                1 / (batch__target__layer_i__sx_sy + 1e-9) - 1
            )  # Size([sum--of--num-boxes, 2])
            batch__target__layer_i__w_h = (
                batch__lbl[:, 3:5] * layer_size
            )  # Size([sum--of--num-boxes, 2])

            # get anchor boxes in layer i
            list__abox__layer_i__pw_ph = (
                self.list__anchor_box__wh[self.list__anchor_box__i_layer == i__layer]
                * layer_scale
            )  # Size([num-anchors-in-layer-i, 2])

            batch__assigned__layer_i__tw_th = torch.log(
                batch__target__layer_i__w_h.unsqueeze(1)
                / list__abox__layer_i__pw_ph.unsqueeze(0)
            )  # Size([sum--of--num-boxes, num-anchors-in-layer-i, 2])

            # calculate iou
            batch__target__layer_i__sx_sy_wh = torch.cat(
                [batch__target__layer_i__sx_sy, batch__target__layer_i__w_h], dim=1
            )  # Size([sum--of--num-boxes, 4])
            batch__abox__layer_i__sx_sy_pw_ph = torch.cat(
                [
                    torch.full_like(
                        list__abox__layer_i__pw_ph, 0.5, dtype=torch.float32
                    ),
                    list__abox__layer_i__pw_ph,
                ],
                axis=1,
            )  # Size([num-anchors-in-layer-i, 4])
            batch__target__sx1_sy1_sx2_sy2 = xcycwh__to__x1y1x2y2(
                batch__target__layer_i__sx_sy_wh, backend="torch"
            ).detach()
            batch__abox__sx1_sy1_sx2_sy2 = xcycwh__to__x1y1x2y2(
                batch__abox__layer_i__sx_sy_pw_ph, backend="torch"
            ).detach()
            mat__iou = box_iou(
                batch__target__sx1_sy1_sx2_sy2, batch__abox__sx1_sy1_sx2_sy2
            ).float()  # Size([sum--of--num-boxes, num-anchors-in-layer-i])

            _batch__xc_yc.append(batch__target__layer_i__xc_yc)
            _batch__w_h.append(batch__target__layer_i__w_h)
            _batch__cx_cy.append(batch__assigned__layer_i__cx_cy)
            _batch__tx_ty.append(batch__assigned__layer_i__tx_ty)
            _batch__tw_th.append(batch__assigned__layer_i__tw_th)
            _batch__iou.append(mat__iou)

        # aggregate
        batch__xc_yc = torch.stack(
            _batch__xc_yc, dim=1
        )  # Size([sum--of--num-boxes, num-layers, 2])
        batch__w_h = torch.stack(
            _batch__w_h, dim=1
        )  # Size([sum--of--num-boxes, num-layers, 2])
        batch__cx_cy = torch.stack(
            _batch__cx_cy, dim=1
        )  # Size([sum--of--num-boxes, num-layers, 2])
        batch__tx_ty = torch.stack(
            _batch__tx_ty, dim=1
        )  # Size([sum--of--num-boxes, num-layers, 2])
        batch__tw_th = torch.stack(
            _batch__tw_th, dim=1
        )  # Size([sum--of--num-boxes, num-layers, num-anchors, 2])
        batch__iou = torch.stack(
            _batch__iou, dim=1
        )  # Size([sum--of--num-boxes, num-layers, num-anchors])

        # assign
        batch__argmatch = []
        for id__img in range(batch_size):
            list__idx__lbl = torch.where(batch__lbl[:, 0] == id__img)
            mat__iou__by__id_img = batch__iou[list__idx__lbl]
            num__lbl, num__layer, num__anchor = mat__iou__by__id_img.shape
            mat__iou__by__id_img = mat__iou__by__id_img.view(-1, num__layer * num__anchor)
            list__idx__row, list__idx__col = linear_sum_assignment(-mat__iou__by__id_img.cpu().numpy())
            for idx__row, idx__col in zip(list__idx__row, list__idx__col):
                idx__lbl = list__idx__lbl[0][idx__row]
                i_layer = idx__col // num__anchor
                i_abox = idx__col % num__anchor
                batch__argmatch.append([i_layer, i_abox])
        batch__argmatch = torch.tensor(batch__argmatch, device=device)




        
        batch__assigned__i_layer = torch.stack(
            [_[0] for _ in batch__argmatch], dim=0
        )  # Size([sum--of--num-boxes])
        batch__assigned__i_abox = torch.stack(
            [_[1] for _ in batch__argmatch], dim=0
        )  # Size([sum--of--num-boxes])
        batch__assigned__iou = torch.stack(
            [m[i] for m, i in zip(batch__iou, batch__argmatch)], dim=0
        )  # list of sum--of--num-boxes
        batch__assigned__cx_cy = torch.stack(
            [b[l] for b, l in zip(batch__cx_cy, batch__assigned__i_layer)], dim=0
        )  # Size([sum--of--num-boxes])
        batch__assigned__tx_ty = torch.stack(
            [b[l] for b, l in zip(batch__tx_ty, batch__assigned__i_layer)], dim=0
        )  # Size([sum--of--num-boxes])
        batch__assigned__tw_th = torch.stack(
            [
                b[l, a]
                for b, l, a in zip(
                    batch__tw_th, batch__assigned__i_layer, batch__assigned__i_abox
                )
            ],
            dim=0,
        )
        batch__target__xc_yc = torch.stack(
            [b[l] for b, l in zip(batch__xc_yc, batch__assigned__i_layer)],
        )
        batch__target__w_h = torch.stack(
            [b[l] for b, l in zip(batch__w_h, batch__assigned__i_layer)],
        )

        return {
            "batch__target__xc_yc": batch__target__xc_yc,
            "batch__target__w_h": batch__target__w_h,
            "batch__assigned__cx_cy": batch__assigned__cx_cy,
            "batch__assigned__tx_ty": batch__assigned__tx_ty,
            "batch__assigned__tw_th": batch__assigned__tw_th,
            "batch__assigned__iou": batch__assigned__iou,
            "batch__assigned__i_layer": batch__assigned__i_layer,
            "batch__assigned__i_abox": batch__assigned__i_abox,
        }
