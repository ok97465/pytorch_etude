#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Train rcnn model for bus truck.

Created on Wed May 25 23:30:47 2022

@author: ok97465
"""
# %% Import
# Standard library imports
import datetime
import os.path as osp

# Third party imports
import torch as th
from torch.multiprocessing import set_sharing_strategy
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, ModelSummary
from pytorch_lightning.loggers import TensorBoardLogger

# Local imports
from dataloader.kaggle_bus_trucks_dataloader import (
    BusTruckDataModuleFasterRCNN,
    bustruck_df2img_info,
)
from ex10_faster_rcnn_model.faster_rcnn_model_bus_truck import BusTruckFasterRCnnModel

set_sharing_strategy("file_system")

# %% Parameter
LR = 0.0005
MAX_EPOCH = 50

BATCH_SIZE = 2
N_WORKER = 0
model_input_size = (224, 224)

device = "mps" if th.backends.mps.is_available() else "gpu"

path_csv = "./data/kaggle_bus_trucks/df.csv"
folder_img = "./data/kaggle_bus_trucks/images"

project_name = "faster_rcnn_model_bus_truck"
folder_log = "logs"
version = 0.001
log_name = datetime.datetime.today().strftime("%y%m%d_%H%M%S") + f"_ver{version}"

# %% Set Model
infos = bustruck_df2img_info(path_csv)[0:8000]
data = BusTruckDataModuleFasterRCNN(
    infos,
    BATCH_SIZE,
    N_WORKER,
    output_size=model_input_size,
    folder_img=folder_img,
)
model = BusTruckFasterRCnnModel(
    lr=LR, model_input_size=model_input_size, batch_size=BATCH_SIZE
)

# %% Set Logger
logger = TensorBoardLogger(save_dir=folder_log, name=project_name, version=log_name)
callbacks = [
    ModelCheckpoint(
        dirpath=osp.join(logger.log_dir, "checkpoint"),
        filename="{epoch}-{val_map:.2f}",
        save_top_k=1,
        monitor="val_map",
        mode="max",
    ),
    ModelSummary(max_depth=-1),
]

# %% Set Trainer
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
trainer.test(model=model, datamodule=data, ckpt_path="best")
