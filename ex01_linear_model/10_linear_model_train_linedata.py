#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Train Perceptron.

Created on Fri Apr 15 15:25:27 2022

@author: ok97465
"""
# %% Import
# Standard library imports
import datetime
import os.path as osp

# Third party imports
import torch
from matplotlib.pyplot import figure
from pytorch_lightning import LightningDataModule, Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, ModelSummary
from pytorch_lightning.loggers import TensorBoardLogger
from torch import arange, randn
from torch.utils.data import DataLoader, TensorDataset

# Local imports
from ex01_linear_model.linear_model import LinearModel


# %% Generate Data
class LineData(LightningDataModule):
    """."""

    def __init__(self, batch_size: int, n_worker: int):
        """."""
        super().__init__()
        self.batch_size = batch_size
        self.n_worker = n_worker

    def setup(self, stage=None):
        """."""
        n_sample = 30
        slope = 1.2
        bias = 2.0

        x_all = arange(n_sample).reshape((-1, 1)).float()
        y_all = x_all * slope + bias + randn((n_sample, 1)) * 3.0
        y_all = y_all.reshape((-1, 1))

        self.train = TensorDataset(x_all, y_all)

    def train_dataloader(self) -> DataLoader:
        """."""
        return DataLoader(
            self.train, batch_size=self.batch_size, num_workers=self.n_worker
        )


# %% Parameter
LR = 0.04
MAX_EPOCH = 100

BATCH_SIZE = 16
N_WORKER = 0

device = "mps" if torch.backends.mps.is_available() else "gpu"
project_name = "linear_model"
folder_log = "logs"
version = 0.001
log_name = datetime.datetime.today().strftime("%y%m%d_%H%M%S") + f"_ver{version}"

# %% Train
data = LineData(BATCH_SIZE, N_WORKER)
model = LinearModel(LR)

logger = TensorBoardLogger(save_dir=folder_log, name=project_name, version=log_name)
callbacks = [
    ModelCheckpoint(
        dirpath=osp.join(logger.log_dir, "checkpoint"),
        filename="{epoch}-{train_loss:.2f}",
        save_top_k=1,
        monitor="train_loss",
        mode="min",
    ),
    ModelSummary(max_depth=-1),
]

trainer = Trainer(
    max_epochs=MAX_EPOCH,
    log_every_n_steps=BATCH_SIZE,
    callbacks=callbacks,
    logger=logger,
    accelerator=device,
    devices=1,
    enable_model_summary=False,
)

trainer.fit(model=model, datamodule=data)

# %% Result
print(model.state_dict())

x_all, y_all = data.train.tensors
y_pred = model(x_all)

x_all = x_all.cpu().detach().numpy().ravel()
y_all = y_all.cpu().detach().numpy().ravel()
y_pred = y_pred.cpu().detach().numpy().ravel()

fig = figure()
ax1 = fig.add_subplot(1, 1, 1)
ax1.scatter(x_all, y_all, label="Data")
ax1.plot(x_all, y_pred, label="Model", c="pink")
ax1.set_xlabel("")
ax1.set_ylabel("")
ax1.set_title("")
ax1.grid(True)
ax1.legend()
fig.tight_layout()
