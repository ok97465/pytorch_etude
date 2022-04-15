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
    BusTruckRoiDataModuleWithSS,
    bustruck_df2img_info,
)
from ex09_fast_rcnn_model.fast_rcnn_model_bus_truck import BusTruckFastRCnnModel

set_sharing_strategy("file_system")

# %% Parameter
LR = 0.0005
MAX_EPOCH = 50

BATCH_SIZE = 2
N_WORKER = 0
model_input_size = (224, 224)

device = "mps" if th.backends.mps.is_available() else "gpu"

ss_param = {
    "scale": 200.0,
    "sigma": 0.8,
    "min_size": 100,
    "iou_th_for_etc": 0.3,
}  # Selective Search parameters

path_csv = "./data/kaggle_bus_trucks/df.csv"
folder_img = "./data/kaggle_bus_trucks/images"

project_name = "fast_rcnn_model_bus_truck"
folder_log = "logs"
version = 0.001
log_name = datetime.datetime.today().strftime("%y%m%d_%H%M%S") + f"_ver{version}"

# %% Set Model
infos = bustruck_df2img_info(path_csv)[0:200]
data = BusTruckRoiDataModuleWithSS(
    infos,
    BATCH_SIZE,
    N_WORKER,
    output_size=model_input_size,
    folder_img=folder_img,
    ss_kwargs=ss_param,
)
model = BusTruckFastRCnnModel(
    lr=LR, model_input_size=model_input_size, batch_size=BATCH_SIZE
)

# %% Set Logger
logger = TensorBoardLogger(save_dir=folder_log, name=project_name, version=log_name)
callbacks = [
    ModelCheckpoint(
        dirpath=osp.join(logger.log_dir, "checkpoint"),
        filename="{epoch}-{valid_acc:.2f}",
        save_top_k=1,
        monitor="valid_acc",
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
