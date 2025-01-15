import os
from tqdm import tqdm
import cv2
import json
import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import ToTensor


class DatasetDetection(Dataset):

    def __init__(self, **kwargs):
        super(DatasetDetection, self).__init__()

        self.path__dir__img = kwargs["path__dir__img"]
        self.path__dir__lbl = kwargs["path__dir__lbl"]
        self.imgsz = kwargs["imgsz"]

        _ = self._load__data()
        self.list__path__file__img = _["list__path__file__img"]
        self.list__path__file__lbl = _["list__path__file__lbl"]

    def _load__data(self):
        list__path__file__img = []
        list__path__file__lbl = []

        for name__file__img in tqdm(
            sorted(os.listdir(self.path__dir__img)), desc="Loading images"
        ):
            path__file__img = os.path.join(self.path__dir__img, name__file__img)
            path__file__lbl = os.path.join(
                self.path__dir__lbl, name__file__img.replace(".jpg", ".json")
            )

            if not os.path.isfile(path__file__lbl):
                continue

            list__path__file__img.append(path__file__img)
            list__path__file__lbl.append(path__file__lbl)

        return {
            "list__path__file__img": list__path__file__img,
            "list__path__file__lbl": list__path__file__lbl,
        }

    def __len__(self):
        return len(self.list__path__file__img)

    def __getitem__(self, idx):
        path__file__img = self.list__path__file__img[idx]
        path__file__lbl = self.list__path__file__lbl[idx]

        img = cv2.imread(path__file__img)
        with open(path__file__lbl, "r") as f:
            dict__result = json.load(f)

        img = cv2.resize(img, (self.imgsz, self.imgsz))

        list__obj__id_class = dict__result["list__obj__id_class"]  # (num-boxes,)
        list__obj__box_xcycwhn = dict__result[
            "list__obj__box_xcycwhn"
        ]  # (num-boxes, 4)

        img = ToTensor()(img.copy())  # HWC [0, 255] -> CHW [0.0, 1.0]
        lbl = torch.from_numpy(
            np.concatenate(
                [
                    np.array(list__obj__box_xcycwhn).reshape(-1, 4),
                    np.array(list__obj__id_class).reshape(-1, 1),
                ],
                axis=1,
            )
        )  # (num-boxes, 5) (xc, yc, w, h, id_class)

        return img, lbl


def collate_fn(batch):
    """
    batch: list of batch_size x tuple (img, lbl).
    """
    list__img, list__lbl = zip(*batch)
    list__id_img = [torch.full((lbl.size(0), 1), i) for i, lbl in enumerate(list__lbl)]

    batch__img = torch.stack(list__img, dim=0)  # (batch-size, 3, imgsz, imgsz)
    batch__lbl = torch.cat(
        [
            torch.cat([id__img, lbl], dim=1)
            for id__img, lbl in zip(list__id_img, list__lbl)
        ],
        dim=0,
    )  # (sum--of--num-boxes, 6) (id__img, xc, yc, w, h, id_class)

    return batch__img, batch__lbl


def get__dataloader(**kwargs):
    dataset = DatasetDetection(**kwargs)
    batch_size = kwargs["batch_size"]
    shuffle = kwargs["shuffle"]
    num_workers = kwargs["num_workers"]

    dataloader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn,
    )

    return dataloader


if __name__ == "__main__":

    dataloader = get__dataloader(
        path__dir__img="/media/laptq/data/workspace/yolov3/datasets/COCO--reformated/val2017/images",
        path__dir__lbl="/media/laptq/data/workspace/yolov3/datasets/COCO--reformated/val2017/labels",
        imgsz=416,
        batch_size=4,
        shuffle=False,
        num_workers=0,
    )

    for img, lbl in dataloader:
        print(img.shape, lbl.shape)
        print(img.dtype, lbl.dtype)
        print(img.min(), img.max())
        break
