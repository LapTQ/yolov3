import yaml
from tqdm import tqdm

from laptq_yolov3.loaders import get__dataloader
from laptq_yolov3.models import YOLOv3
from laptq_yolov3.losses import LossYOLOv3
from laptq_pyutils.log import load_logger

import torch
from torch.optim import SGD
from torch.optim.lr_scheduler import ReduceLROnPlateau


LOGGER = load_logger()


class Trainer:

    def __init__(self, **kwargs):
        self.dataloader__train = kwargs["dataloader__train"]
        self.model = kwargs["model"]
        self.optimizer = kwargs["optimizer"]
        self.criterion = kwargs["criterion"]
        self.device = kwargs["device"]
        self.scheduler = kwargs["scheduler"]

        self.model.to(self.device)

    def train(self, **kwargs):

        num__epoch = kwargs["num__epoch"]
        num__step_running = kwargs["num__step_running"]

        for i__epoch in range(num__epoch):

            _ = self._train__batch(
                num__step_running=num__step_running,
            )
            loss__train = _["loss"]

            _ = self._val_batch()
            loss__val = _["loss"]

            LOGGER.bind(classname=self.__class__.__name__).info(
                "Epoch {}/{}:\n\tloss__train: {}, loss__val: {}, lr={}".format(
                    i__epoch,
                    num__epoch - 1,
                    loss__train,
                    loss__val,
                    self.scheduler.get_last_lr(),
                )
            )

            self.scheduler.step(loss__val)

    def _train__batch(self, **kwargs):

        num__step_running = kwargs["num__step_running"]

        loss__running = 0.0
        loss__epoch = 0.0

        self.model.train()

        pbar = tqdm(enumerate(self.dataloader__train), desc="Training")
        for i__batch, batch in pbar:
            batch__img, batch__lbl = batch

            batch__img = batch__img.to(self.device)
            batch__lbl = batch__lbl.to(self.device)

            self.optimizer.zero_grad()

            batch__output = self.model(batch__img)

            loss = self.criterion(batch__output, batch__lbl)
            loss.backward()
            self.optimizer.step()

            loss__running += loss.item()
            loss__epoch += loss.item()

            if i__batch % num__step_running == 0:
                loss__running /= num__step_running
                pbar.set_postfix(
                    batch="{}/{}".format(i__batch, len(self.dataloader__train) - 1),
                    loss=loss__running,
                )
                loss__running = 0.0

        loss__epoch /= len(self.dataloader__train)

        return {
            "loss": loss__epoch,
        }

    def _val__batch(self, **kwargs):

        num__step_running = kwargs["num__step_running"]

        loss__running = 0.0

        self.model.eval()

        with torch.no_grad():
            pbar = tqdm(enumerate(self.dataloader__val), desc="Validating")
            for i__batch, batch in pbar:
                batch__img, batch__lbl = batch

                batch__img = batch__img.to(self.device)
                batch__lbl = batch__lbl.to(self.device)

                batch__output = self.model(batch__img)

                loss = self.criterion(batch__output, batch__lbl)

                loss__running += loss.item()

                if i__batch % num__step_running == 0:
                    loss__running /= num__step_running
                    pbar.set_postfix(
                        batch="{}/{}".format(i__batch, len(self.dataloader__val) - 1),
                        loss=loss__running,
                    )
                    loss__running = 0.0

        loss__val = loss__running / len(self.dataloader__val)

        return {
            "loss": loss__val,
        }


if __name__ == "__main__":

    dataloader__train = get__dataloader(
        path__dir__img="/media/laptq/data/workspace/yolov3/datasets/COCO--reformated/val2017/images",
        path__dir__lbl="/media/laptq/data/workspace/yolov3/datasets/COCO--reformated/val2017/labels",
        imgsz=416,
        batch_size=4,
        shuffle=True,
        num_workers=0,
    )

    with open(
        "/media/laptq/data/workspace/yolov3/datasets/COCO--reformated/train2017/anchor-boxes.yaml",
        "r",
    ) as f:
        list__anchor_box__wh = yaml.load(f, Loader=yaml.FullLoader)
    list__anchor_box__i_layer = [0, 0, 0, 1, 1, 1, 2, 2, 2] # 
    assert len(list__anchor_box__wh) == len(list__anchor_box__i_layer)

    model = YOLOv3(
        num__anchors=3,
        num__classes=90,
        list__anchor_box__wh=list__anchor_box__wh,
        list__anchor_box__i_layer=list__anchor_box__i_layer,
    )

    optimizer = SGD(
        model.parameters(),
        lr=1e-3,
        momentum=0.9,
        nesterov=True,
    )

    criterion = LossYOLOv3(
        model=model,
    )

    device = torch.device("cuda:0")

    scheduler = ReduceLROnPlateau(
        optimizer=optimizer, mode="min", factor=0.1, patience=10
    )

    trainer = Trainer(
        dataloader__train=dataloader__train,
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        scheduler=scheduler,
    )

    trainer.train(
        num__epoch=10,
        num__step_running=1,
    )
