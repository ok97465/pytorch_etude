#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Show the result of faster R-CNN.

Created on Fri May 27 22:17:53 2022

@author: ok97465
"""
# %% Import
# Standard library imports

# Third party imports
from matplotlib.patches import Rectangle
from matplotlib.pyplot import figure
from torch import tensor
from torch.autograd.grad_mode import no_grad
from torchvision.io import read_image
from torchvision.ops import nms

# Local imports
from dataloader.kaggle_bus_trucks_dataloader import (
    TARGET2LABEL,
    BusTruckDataModuleFasterRCNN,
    bustruck_df2img_info,
)
from ex10_faster_rcnn_model.faster_rcnn_model_bus_truck import BusTruckFasterRCnnModel

# %% Parameter
path_model_ckpt = (
    "./logs/faster_rcnn_model_bus_truck/220610_212010_ver0.001/checkpoint/"
    "epoch=4-val_map=0.78.ckpt"
)
path_csv = "./data/kaggle_bus_trucks/df.csv"
folder_img = "./data/kaggle_bus_trucks/images"
model_input_size = (224, 224)

# %% Load data
infos = bustruck_df2img_info(path_csv)[:30]
data = BusTruckDataModuleFasterRCNN(
    infos,
    1,
    0,
    output_size=model_input_size,
    folder_img=folder_img,
)
data.setup()

# %% Load Model
model = BusTruckFasterRCnnModel(model_input_size=model_input_size).load_from_checkpoint(
    path_model_ckpt
)
model.eval()

# %% Detection & Classification
for sample in iter(data.train_dataloader()):
    image_id = sample[1][0]["image_id"]
    # Fowarding
    with no_grad():
        preds = model(sample)[0]

    # %% Plot Result
    path_img = f"{folder_img}/{image_id}.jpg"
    img = read_image(path_img).permute(1, 2, 0)
    n_y, n_x = img.shape[:2]
    scale_box = tensor([n_x, n_y, n_x, n_y]) / 224

    # Draw image
    fig = figure()
    ax1 = fig.add_subplot(1, 1, 1)
    ax1.imshow(img)
    ax1.set_title(f"Faster RCNN Result of {image_id}")
    ax1.grid(False)

    # Apply NMS
    idx_nms = nms(preds["boxes"], preds["scores"], 0.05)
    preds_nms = {k: v[idx_nms] for k, v in preds.items()}

    # Prediction box
    annotation = "Prediction"
    for box, label_idx, score in zip(
        preds_nms["boxes"], preds_nms["labels"], preds_nms["scores"]
    ):
        box *= scale_box
        ax1.add_patch(
            Rectangle(
                box[0:2],
                *(box[2:4] - box[0:2]),
                linewidth=2,
                edgecolor="r",
                fill=False,
                label=annotation,
            )
        )
        ax1.text(
            box[0],
            box[1] - 1,
            f"{TARGET2LABEL[label_idx.item()]}({score:.2f})",
            color="r",
        )
        annotation = None

    # Exact box
    annotation = "Exact"
    exact = sample[1][0]
    for box, label_idx in zip(exact["boxes"], exact["labels"]):
        box *= scale_box
        ax1.add_patch(
            Rectangle(
                box[0:2],
                *(box[2:4] - box[0:2]),
                linewidth=2,
                edgecolor="b",
                fill=False,
                label=annotation,
            )
        )
        ax1.text(
            box[0],
            box[1] - 1,
            TARGET2LABEL[label_idx.item()],
            color="b",
        )
        annotation = None

    ax1.legend()
    fig.tight_layout()
