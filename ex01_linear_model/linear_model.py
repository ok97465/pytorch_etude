#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Perceptron Model.

Created on Fri Apr 15 16:00:16 2022

@author: ok97465
"""
# Third party imports
from pytorch_lightning import LightningModule
from pytorch_lightning.utilities.types import STEP_OUTPUT
from torch import Tensor, zeros
from torch.nn import Linear, ModuleList, MSELoss
from torch.optim import Adam
from torch.optim.optimizer import Optimizer


class LinearModel(LightningModule):
    """Perceptron Model."""

    def __init__(self, lr: float = 0.001):
        """."""
        super().__init__()
        self.lr = lr

        self.example_input_array = zeros(1, 1)  # For model summary
        self.module_list = ModuleList([Linear(1, 1)])
        self.calc_loss = MSELoss()

    def forward(self, x: Tensor) -> Tensor:
        """Forward."""
        for m in self.module_list:
            x = m(x)
        return x

    def training_step(self, batch: list[Tensor], batch_idx: int) -> STEP_OUTPUT:
        """Step."""
        x, y = batch
        logits = self(x)
        loss = self.calc_loss(logits, y)

        self.log("train_loss", loss, prog_bar=True)
        return loss

    def on_train_epoch_end(self):
        """End step."""
        pass

    def configure_optimizers(self) -> Optimizer:
        """."""
        optimizer = Adam(self.parameters(), lr=self.lr)
        return optimizer
