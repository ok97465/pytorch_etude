#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Train denoise autoencoder model for MNIST.

Created on Sat Apr 23 08:22:07 2022

@author: ok97465
"""
# %% Import
# Standard library imports
import datetime
import os.path as osp

# Third party imports
import torch as th
from numpy import newaxis
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, ModelSummary
from pytorch_lightning.loggers import TensorBoardLogger

# Local imports
from dataloader.mnist_dataloader import MnistData
from ex04_denoise_AE_mnist.denoise_autoencoder_model_mnist import MnistDenoiseAEModel

# %% Parameter
LR = 0.001
MAX_EPOCH = 50

BATCH_SIZE = 200
N_WORKER = 0

device = "mps" if th.backends.mps.is_available() else "gpu"
project_name = "denoise_autoencoder_mnist"
folder_log = "logs"
version = 0.001
log_name = datetime.datetime.today().strftime("%y%m%d_%H%M%S") + f"_ver{version}"

# %% Set Model
data = MnistData(BATCH_SIZE, N_WORKER)
data.setup()
model = MnistDenoiseAEModel(LR)

# %% Set Logger
logger = TensorBoardLogger(save_dir=folder_log, name=project_name, version=log_name)
callbacks = [
    ModelCheckpoint(
        dirpath=osp.join(logger.log_dir, "checkpoint"),
        filename="{epoch}-{valid_loss:.7f}",
        save_top_k=1,
        monitor="valid_loss",
        mode="min",
    ),
    ModelSummary(max_depth=-1),
]

# %% Set Trainer
gpus = 1 if th.cuda.is_available() else None
trainer = Trainer(
    max_epochs=MAX_EPOCH,
    log_every_n_steps=len(data.train_dataloader()),
    callbacks=callbacks,
    logger=logger,
    accelerator=device,
    devices=1,
    enable_model_summary=False,
)

trainer.fit(model=model, datamodule=data)
trainer.test(model=model, datamodule=data, ckpt_path="best")

# %% Add model graph to tensorboard
x = data.test.data.float()
x1 = x[0][newaxis, newaxis, :, :]
logger.experiment.add_graph(model, x1)
