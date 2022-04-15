#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""CNN model for mnist.

Created on Sat Apr 23 07:59:29 2022

@author: ok97465
"""
# %% Import
# Standard library imports
from collections import defaultdict
from typing import Optional

# Third party imports
import matplotlib.pyplot as plt
import torch as th
from pytorch_lightning import LightningModule
from pytorch_lightning.utilities.types import STEP_OUTPUT
from torch import Tensor, argmax, zeros
from torch.nn import (
    Conv2d,
    CrossEntropyLoss,
    Dropout,
    Flatten,
    Linear,
    MaxPool2d,
    Module,
    ReLU,
    Sequential,
    Softmax,
)
from torch.optim import Adam
from torch.optim.optimizer import Optimizer
from torchmetrics.classification.accuracy import Accuracy
from torchmetrics.functional import confusion_matrix

# Local imports
from helper.helper import plot_confusion_matrix


def conv_block(n_in: int, n_out: int) -> Module:
    """."""
    model = Sequential()
    model.add_module("dropout", Dropout(0.2))
    model.add_module("conv2d", Conv2d(n_in, n_out, kernel_size=5, padding=2))
    model.add_module("relu", ReLU())
    # model.add_module("batch_norm", BatchNorm2d(n_out))
    model.add_module("maxpool", MaxPool2d(2))
    return model


class MnistCnnModel(LightningModule):
    """CNN model for MNIST."""

    def __init__(self, lr: float = 0.001):
        """."""
        super().__init__()
        self.lr = lr
        self.n_class = 10
        self.example_input_array = zeros(10, 1, 28, 28)  # For model summary
        self.labels = [str(idx) for idx in range(self.n_class)]

        self.model = Sequential()
        # Stage1
        self.model.add_module("conv_block1", conv_block(1, 32))
        self.model.add_module("conv_block2", conv_block(32, 64))

        # Stage2
        self.model.add_module("flatten", Flatten())
        self.model.add_module("fc1", Linear(3136, 1024))
        self.model.add_module("dropout", Dropout(p=0.2))
        self.model.add_module("fc2", Linear(1024, 10))
        self.model.add_module("relu3", ReLU())
        self.model.add_module("softmax", Softmax(dim=1))

        self.calc_loss = CrossEntropyLoss()

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

        self.intermediate = defaultdict(list)

    def forward(self, x: Tensor) -> Tensor:
        """Forward."""
        pred = self.model(x)
        return pred

    def _step(self, batch: list[Tensor], batch_idx: int, data_name: str) -> STEP_OUTPUT:
        """Kernel of step."""
        x, y = batch
        logits = self.forward(x)

        loss = self.calc_loss(logits, y)
        self.log(f"{data_name}_loss", loss, prog_bar=True)

        preds = argmax(logits, dim=1)
        accuracy = self.accuracy[data_name]
        accuracy.update(preds, y)

        self.intermediate["loss"].append(loss)
        self.intermediate["preds"].append(preds)
        self.intermediate["targets"].append(y)
        self.intermediate["data"].append(x)
        self.intermediate["logits"].append(logits)

        return loss

    def _epoch_end(self, data_name: str):
        """Kenel of epoch_end."""
        accuracy = self.accuracy[data_name]
        self.log(f"{data_name}_acc", accuracy.compute(), prog_bar=True)
        accuracy.reset()

    def training_step(self, batch: list[Tensor], batch_idx: int) -> STEP_OUTPUT:
        """Training step."""
        return self._step(batch, batch_idx, "train")

    def validation_step(self, batch: list[Tensor], batch_idx: int) -> STEP_OUTPUT:
        """Validation step."""
        return self._step(batch, batch_idx, "valid")

    def test_step(self, batch: list[Tensor], batch_idx: int) -> STEP_OUTPUT:
        """Test step."""
        data_name = "test"
        loss = self._step(batch, batch_idx, data_name)
        self._epoch_end(data_name)

        return loss

    def on_training_epoch_end(self):
        """."""
        self._epoch_end("train")
        self.intermediate.clear()

    def on_validation_epoch_end(self):
        """."""
        self._epoch_end("valid")

        data = th.cat(self.intermediate["data"])
        preds = th.cat(self.intermediate["preds"])
        targets = th.cat(self.intermediate["targets"])
        logits = th.cat(self.intermediate["logits"])

        # Confusion matrix
        cm = confusion_matrix(
            preds, targets, task="multiclass", num_classes=self.n_class
        )
        cm = cm.cpu().detach().numpy()
        fig, ax = plot_confusion_matrix(cm, self.labels, cmap_log_scale=True)
        plt.close(fig)
        self.logger.experiment.add_figure("Confusion matrix", fig, self.current_epoch)

        # Embedding
        n_dis = data.shape[0]
        x_ = data[:n_dis, :, :, :]
        y_ = logits
        metadata = [str(t.item()) for t in targets]

        self.logger.experiment.add_embedding(
            y_, metadata=metadata, label_img=x_, global_step=self.current_epoch
        )
        self.intermediate.clear()

    def configure_optimizers(self) -> Optimizer:
        """Configure optimizer."""
        optimizer = Adam(self.parameters(), lr=self.lr)
        return optimizer
