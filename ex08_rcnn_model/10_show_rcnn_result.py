#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Show the result of R-CNN.

Created on Fri May 27 22:17:53 2022

@author: ok97465
"""
# %% Import
# Standard library imports
from typing import Optional

# Third party imports
from matplotlib.patches import Rectangle
from matplotlib.pyplot import figure
from torch import nonzero, tensor
from torch.autograd.grad_mode import no_grad
from torch.nn.functional import softmax
from torchvision.io import read_image
from torchvision.ops import nms

# Local imports
from dataloader.kaggle_bus_trucks_dataloader import (
    LABEL2TARGET,
    TARGET2LABEL,
    BusTruckCroppedDataModuleWithSS,
    bustruck_df2img_info,
)
from ex08_rcnn_model.rcnn_model_bus_truck import BusTruckRCnnModel

# %% Parameter
path_model_ckpt = (
    "./logs/rcnn_model_bus_truck/220528_130939_ver0.001/checkpoint/"
    "epoch=7-valid_acc=0.73.ckpt"
)
path_csv = "./data/kaggle_bus_trucks/df.csv"
folder_img = "./data/kaggle_bus_trucks/images"
model_input_size = (224, 224)

ss_param = {
    "scale": 200.0,
    "sigma": 0.8,
    "min_size": 100,
    "iou_th_for_etc": 0.3,
}  # Selective Search parameters


# %% Load data
infos = bustruck_df2img_info(path_csv)[100:110]
data = BusTruckCroppedDataModuleWithSS(
    infos,
    1,
    0,
    output_size=model_input_size,
    folder_img=folder_img,
    ss_kwargs=ss_param,
)
data.setup()

# %% Load Model
model = BusTruckRCnnModel(model_input_size=model_input_size).load_from_checkpoint(
    path_model_ckpt
)
model.eval()

# %% Detection & Classification
for sample in iter(data.train_dataloader()):
    with no_grad():
        scores, deltas = model(sample["imgs"])
        probs = softmax(scores, -1)
        confidences, pred_tg_idx = probs.max(-1)

    # %% Plot Result
    path_img = f"{folder_img}/{sample['image_id'][0]}.jpg"
    img = read_image(path_img).permute(1, 2, 0)
    n_y, n_x = img.shape[:2]
    ratio2coordi = tensor([n_x, n_y, n_x, n_y])

    # Draw image
    fig = figure()
    ax1 = fig.add_subplot(1, 1, 1)
    ax1.imshow(img)
    ax1.set_title("RCNN Result")
    ax1.grid(False)

    # Get info of best score
    idx_bus_truck = nonzero(pred_tg_idx != LABEL2TARGET["Etc"]).ravel()
    if len(idx_bus_truck) < 1:
        ax1.set_title(
            f"RCNN Result of {sample['image_id'][0]}", fontdict={"color": "r"}
        )
        continue

    deltas_bus_truck = deltas[idx_bus_truck] * ratio2coordi
    confidences_bus_truck = confidences[idx_bus_truck]
    pred_tg_idx_bus_truck = pred_tg_idx[idx_bus_truck]
    box_bus_truck = sample["box_ratio"][idx_bus_truck] * ratio2coordi

    pred_box_bus_truck = box_bus_truck + deltas_bus_truck
    idx_nms = nms(pred_box_bus_truck, confidences_bus_truck, 0.05)

    if len(idx_nms) < 1:
        raise RuntimeError("Failed to NMS")
        continue

    label1: Optional[str] = "Box of RCNN"
    label2: Optional[str] = "Box of SS"
    for idx in idx_nms:
        # Draw box of rcnn
        coordinate = pred_box_bus_truck[idx]
        ax1.add_patch(
            Rectangle(
                coordinate[0:2],
                *(coordinate[2:4] - coordinate[0:2]),
                linewidth=2,
                edgecolor="r",
                fill=False,
                label=label1,
            )
        )
        ax1.text(
            coordinate[0],
            coordinate[1],
            TARGET2LABEL[pred_tg_idx_bus_truck[idx].item()],
            color="r",
        )

        # Draw box of selective search
        ss_coordinate = box_bus_truck[idx]
        ax1.add_patch(
            Rectangle(
                ss_coordinate[0:2],
                *(ss_coordinate[2:4] - ss_coordinate[0:2]),
                linewidth=2,
                edgecolor="g",
                fill=False,
                label=label2,
            )
        )
        label1 = None
        label2 = None
    ax1.legend()
    fig.tight_layout()
