# -*- coding: utf-8 -*-
"""Linear model for mnist.

Created on Sat Apr 16 08:12:07 2022

@author: ok97465
"""
# %% Import
# Standard library imports
from collections import defaultdict

# Third party imports
import matplotlib.pyplot as plt
import torch as th
from pytorch_lightning import LightningModule
from pytorch_lightning.utilities.types import STEP_OUTPUT
from torch import Tensor, argmax, zeros
from torch.nn import (
    CrossEntropyLoss,
    Dropout,
    Flatten,
    Linear,
    ModuleList,
    ReLU,
    Softmax,
)
from torch.optim import Adam
from torch.optim.optimizer import Optimizer
from torchmetrics.classification.accuracy import Accuracy
from torchmetrics.functional import confusion_matrix

# Local imports
from helper.helper import plot_confusion_matrix


class MnistLinearModel(LightningModule):
    """Linear model hand written data from MNIST."""

    def __init__(self, lr: float = 0.001) -> None:
        """."""
        super().__init__()
        self.lr = lr

        self.example_input_array = zeros(10, 1, 28, 28)  # For model summary
        self.n_class = 10
        self.labels = [str(idx) for idx in range(self.n_class)]
        n_element_img = 28 * 28
        n_hid1 = 64
        n_hid2 = 32
        n_hid3 = 16

        self.module_list = ModuleList(
            [
                Flatten(),
                Linear(n_element_img, n_hid1),
                ReLU(),
                Dropout(0.1),
                Linear(n_hid1, n_hid2),
                ReLU(),
                Dropout(0.1),
                Linear(n_hid2, n_hid3),
                ReLU(),
                Dropout(0.1),
                Linear(n_hid3, self.n_class),
                Softmax(dim=1),
            ]
        )

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
        for m in self.module_list:
            x = m(x)
        return x

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

        # Add Weight histogram
        for name, params in self.named_parameters():
            self.logger.experiment.add_histogram(name, params, self.current_epoch)

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
