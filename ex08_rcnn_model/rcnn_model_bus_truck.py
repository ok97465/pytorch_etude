#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""RCNN model for kaggle bus truck.

Created on Wed May 25 19:19:04 2022

@author: ok97465
"""
# %% Import
# Standard library imports
from typing import Optional

# Third party imports
from pytorch_lightning import LightningModule
from torch import Tensor, nonzero, zeros
from torch.nn import CrossEntropyLoss, L1Loss, Linear, ReLU, Sequential, Tanh
from torch.optim import SGD
from torch.optim.optimizer import Optimizer
from torchmetrics.classification.accuracy import Accuracy
from torchvision.models import vgg16

# Local imports
from dataloader.kaggle_bus_trucks_dataloader import LABEL2TARGET


class BusTruckRCnnModel(LightningModule):
    """RCNN model for Kaggle BusTruck."""

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
        self.example_input_array = zeros(10, 3, self.input_size[0], self.input_size[1])
        self.w_for_delta_in_loss = 10.0
        feature_dim = 25088

        self.backbone = vgg16(pretrained=True)
        self.backbone.classifier = Sequential()
        for param in self.backbone.parameters():
            param.requires_grad = False
        self.backbone.eval()

        self.model_class_score = Linear(feature_dim, self.n_class)
        self.model_box_delta = Sequential(
            Linear(feature_dim, 512), ReLU(), Linear(512, 4), Tanh()
        )

        self.cross_entropy = CrossEntropyLoss()
        self.l1loss = L1Loss()

        # Metric
        param_acc = {
            "task": "multiclass",
            "threshold": 0.5,
            "average": "macro",
            "num_classes": self.n_class,
        }
        self.train_acc = Accuracy(**param_acc)
        self.val_acc = Accuracy(**param_acc)
        self.test_acc = Accuracy(**param_acc)

        self.accuracy = {
            "train": self.train_acc,
            "valid": self.val_acc,
            "test": self.test_acc,
        }

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        """Forward."""
        feat = self.backbone(x)
        class_score = self.model_class_score(feat)
        delta = self.model_box_delta(feat)
        return class_score, delta

    def calc_loss(self, pred_score, tg_idx, pred_delta, exact_delta):
        """."""
        classfier_loss = self.cross_entropy(pred_score, tg_idx)
        idx_interest = nonzero(tg_idx != LABEL2TARGET["Etc"])
        _pred_delta = pred_delta[idx_interest]
        _exact_delta = exact_delta[idx_interest]

        if len(idx_interest) > 0:
            delta_loss = self.l1loss(_pred_delta, _exact_delta)
        else:
            delta_loss = 0

        return classfier_loss + self.w_for_delta_in_loss * delta_loss

    def _step(self, batch: dict, batch_idx: int, data_name: str) -> Tensor:
        """Kernel of step."""
        imgs = batch["imgs"]
        exact_deltas = batch["deltas"]
        exact_tg_idx = batch["tg_idx"]
        pred_scores, pred_deltas = self.forward(imgs)

        loss = self.calc_loss(pred_scores, exact_tg_idx, pred_deltas, exact_deltas)
        self.log(f"{data_name}_loss", loss, prog_bar=True, batch_size=self.batch_size)

        accuracy = self.accuracy[data_name]
        accuracy.update(pred_scores, exact_tg_idx)

        return loss

    def _epoch_end(self, data_name: str):
        """Kenel of epoch_end."""
        accuracy = self.accuracy[data_name]
        self.log(
            f"{data_name}_acc",
            accuracy.compute(),
            prog_bar=True,
            batch_size=self.batch_size,
        )
        accuracy.reset()

    def training_step(self, batch: dict, batch_idx: int) -> Tensor:
        """Training step."""
        return self._step(batch, batch_idx, "train")

    def validation_step(self, batch: dict, batch_idx: int) -> Tensor:
        """Validation step."""
        return self._step(batch, batch_idx, "valid")

    def test_step(self, batch: dict, batch_idx: int) -> Tensor:
        """Test step."""
        data_name = "test"
        loss = self._step(batch, batch_idx, data_name)
        self._epoch_end(data_name)

        return loss

    def on_training_epoch_end(self):
        """."""
        self._epoch_end("train")

    def on_validation_epoch_end(self):
        """."""
        self._epoch_end("valid")

    def configure_optimizers(self) -> Optimizer:
        """Configure optimizer."""
        optimizer = SGD(self.parameters(), lr=self.lr)
        return optimizer
