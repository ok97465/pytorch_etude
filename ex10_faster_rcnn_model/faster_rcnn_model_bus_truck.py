#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Faster RCNN model for kaggle bus truck.

Created on Tue Jun  7 23:31:51 2022

@author: ok97465
"""
# %% Import
# Standard library imports
# Third party imports
from pytorch_lightning import LightningModule
from torch import Tensor, zeros
from torch.optim import SGD
from torch.optim.optimizer import Optimizer
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

# Local imports
from dataloader.kaggle_bus_trucks_dataloader import LABEL2TARGET


class BusTruckFasterRCnnModel(LightningModule):
    """Faster RCNN model for Kaggle BusTruck."""

    def __init__(
        self,
        lr: float = 0.001,
        model_input_size: tuple[int, int] = (224, 224),
        batch_size: int = 2,
    ):
        """."""
        super().__init__()
        self.batch_size = batch_size
        self.lr = lr
        self.n_class = len(LABEL2TARGET)
        self.input_size: tuple[int, int] = model_input_size
        # self.example_input_array = (
        #     zeros(3, self.input_size[0], self.input_size[1]),
        #     {"boxes": zeros(2, 4), "labels": zeros(2).long()},
        # )
        self.model = fasterrcnn_resnet50_fpn(pretrained=True)
        in_features = self.model.roi_heads.box_predictor.cls_score.in_features
        self.model.roi_heads.box_predictor = FastRCNNPredictor(
            in_features, self.n_class
        )

        self.map_metric = MeanAveragePrecision()

    def forward(self, inputs) -> Tensor:
        """Forward."""
        imgs, infos = inputs

        self.model.eval()
        output = self.model(imgs)

        return output

    def training_step(self, batch: dict, batch_idx: int) -> Tensor:
        """Training step."""
        imgs, infos = batch
        loss_dict = self.model(imgs, infos)
        losses = sum(loss for loss in loss_dict.values())

        batch_size = len(batch[0])
        self.log_dict(loss_dict, batch_size=batch_size)
        self.log("train_loss", losses, batch_size=batch_size)

        return losses

    def validation_step(self, batch: dict, batch_idx: int):
        """Validation step."""
        imgs, infos = batch
        preds = self.model(imgs)

        self.map_metric.update(preds, infos)

    def on_validation_epoch_end(self):
        """."""
        ret_map = self.map_metric.compute()
        self.log("val_map", ret_map["map_50"])
        self.map_metric.reset()

    def test_step(self, batch: dict, batch_idx: int) -> Tensor:
        """Test step."""
        imgs, infos = batch
        preds = self.model(imgs)

        self.map_metric(preds, infos)
        ret_map = self.map_metric.compute()

        self.log("test_map", ret_map["map_50"])

    def configure_optimizers(self) -> Optimizer:
        """Configure optimizer."""
        optimizer = SGD(
            self.parameters(), lr=self.lr, momentum=0.9, weight_decay=0.0005
        )
        return optimizer
