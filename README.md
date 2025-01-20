

- thiết kế mạng:
    - backbone: darknet
    - neck: FPN
    - head: 3*(4+1+80)
- dataloader:
    - ảnh: BxCxHxW
    - label: nboxes x [img, class, x, y, w, h]

- loss:
    - gán GT với anchor: 1-1 (trên cả 3 tầng)
        - rescale GT box về tọa độ của 1 tầng => xem tâm GT box đó thuộc cell nào trong grid => tính offset của tâm box với vị trí của cell đó
        - tính iou của GT box với anchor (tâm anchor đặt tâm cell, không phải tâm box).
        - Chỉ xét iou của GT box với 3 anchor (x3=9 chỏ 3 tầng) tại cell chứa tâm GT thôi, chứ không xét tất cả anchor rải rác khắp chiều ngang dọc.
    - với những anchor được gán: tính 3 loss: objness, iou, cls
        - objness: BCE (logit + sigmoid)
            - gt: iou (tại sao kp 1?)
            - pred
        - iou: 1 - iou
        - cls: BCE (logit + sigmoid)
            - gt: one-hot
            - pred
    - với những anchor không được gán: chỉ tính loss: objness


NMS: chỉ cần ở lúc detect.




This new network is much more powerful than Darknet-
19 but still more efficient than ResNet-101 or ResNet-152

We use multi-scale training, lots of data
augmentation, batch normalization, all the standard stuff.