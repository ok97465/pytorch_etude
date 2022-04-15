#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Autoencoder model for mnist.

Created on Sat Apr 23 07:59:29 2022

@author: ok97465
"""
# %% Import
# Standard library imports
from collections import defaultdict
from typing import Optional

# Third party imports
import torch as th
from pytorch_lightning import LightningModule
from pytorch_lightning.utilities.types import STEP_OUTPUT
from torch import Tensor, zeros
from torch.autograd.grad_mode import no_grad
from torch.nn import (
    Conv2d,
    ConvTranspose2d,
    MaxPool2d,
    MSELoss,
    ReLU,
    Sequential,
    Sigmoid,
)
from torch.optim import Adam
from torch.optim.optimizer import Optimizer
from torchvision.utils import make_grid


class MnistDenoiseAEModel(LightningModule):
    """Denoise Autoencoder model for MNIST."""

    def __init__(self, lr: float = 0.001):
        """."""
        super().__init__()
        self.lr = lr
        self.sample_imgs: Tensor = Tensor()
        self.example_input_array = zeros(10, 1, 28, 28)  # For model summary

        # Encoder
        self.model_en = Sequential()
        # Encoder Stage1
        self.model_en.add_module(
            "conv1", Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1)
        )
        self.model_en.add_module("relu1", ReLU())
        self.model_en.add_module("pool1", MaxPool2d(kernel_size=2))

        # Encoder Stage2
        self.model_en.add_module(
            "conv2", Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1)
        )
        self.model_en.add_module("relu2", ReLU())
        self.model_en.add_module("pool2", MaxPool2d(kernel_size=2))

        # Decoder
        self.model_de = Sequential()
        self.model_de.add_module(
            "deconv1",
            ConvTranspose2d(in_channels=32, out_channels=32, kernel_size=2, stride=2),
        )
        self.model_de.add_module("relu1", ReLU())
        self.model_de.add_module(
            "deconv2",
            ConvTranspose2d(in_channels=32, out_channels=32, kernel_size=2, stride=2),
        )
        self.model_de.add_module("relu2", ReLU())

        self.model_de.add_module(
            "conv1", Conv2d(in_channels=32, out_channels=1, kernel_size=3, padding=1)
        )
        self.model_de.add_module("sigmode", Sigmoid())

        self.calc_loss = MSELoss()

        self.intermediate = defaultdict(list)

    def forward(self, x: Tensor) -> Tensor:
        """Forward."""
        x_en = self.model_en(x)
        x_de = self.model_de(x_en)
        return x_de

    def _step(self, batch: list[Tensor], batch_idx: int, data_name: str) -> STEP_OUTPUT:
        """Kernel of step."""
        x, y = batch
        logits = self.forward(x)

        loss = self.calc_loss(logits, x)
        self.log(f"{data_name}_loss", loss, prog_bar=True)

        self.intermediate["loss"].append(loss)
        self.intermediate["data"].append(x)

        return loss

    def _epoch_end(self, data_name: str):
        """Kenel of epoch_end."""
        pass

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

        # Send original sample images to tensorboard
        n_sample = 4
        if self.current_epoch == 0:
            data = th.cat(self.intermediate["data"])[: n_sample + 1]
            self.sample_imgs = data.clone().detach()
            grid = make_grid(self.sample_imgs)
            self.logger.experiment.add_image("input", grid, 0)

        # Send the result of sample images to tensorboard
        with no_grad():
            ret = self.forward(self.sample_imgs)
            grid = make_grid(ret)
            self.logger.experiment.add_image("output", grid, self.current_epoch)

        self.intermediate.clear()

    def configure_optimizers(self) -> Optimizer:
        """Configure optimizer."""
        optimizer = Adam(self.parameters(), lr=self.lr)
        return optimizer
