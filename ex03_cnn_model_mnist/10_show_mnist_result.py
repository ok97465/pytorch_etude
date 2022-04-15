# -*- coding: utf-8 -*-
"""Show the result of MNIST.

Created on Sat Apr 16 11:18:07 2022

@author: ok97465
"""
# %% Import
# Third party imports
import torch
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.pyplot import subplots
from numpy import logical_and, ndarray
from numpy.random import choice
from torch import argmax, where
from torch.nn import Sequential
from torchmetrics.functional import confusion_matrix

# Local imports
from dataloader.mnist_dataloader import MnistData
from ex03_cnn_model_mnist.cnn_model_mnist import MnistCnnModel
from helper.helper import plot_confusion_matrix

# %% Parameters
path_data = "./data"
path_model_ckpt = (
    "./logs/cnn_model_mnist/230513_093434_ver0.001/checkpoint/"
    "epoch=24-valid_acc=0.99.ckpt"
)
n_class = 10
n_img_sample = 10000

# %% Data Load
device = torch.device("mps" if torch.backends.mps.is_available() else "gpu")
data_loader = MnistData(n_img_sample)
data_loader.setup()

data_sample, targets_sample = next(iter(data_loader.test_dataloader()))

data_sample = data_sample.to(device)
targets_sample = targets_sample.to(device)

labels = [str(idx) for idx in range(10)]

# %% Load Model
model = MnistCnnModel().load_from_checkpoint(path_model_ckpt)
# model.to(device)
model.eval()

# %% Prediction
probabilitys = model.forward(data_sample)
predicts = argmax(probabilitys, dim=1)

# %% To CPU
predicts = predicts.cpu()
data_sample = data_sample.cpu()
targets_sample = targets_sample.cpu()


# %% Plot image
def plot_number(
    idxes: ndarray, size: tuple[int, int] = (2, 4)
) -> tuple[Figure, list[Axes]]:
    """Plot number."""
    fig, axes = subplots(*size, sharex=True, sharey=True)
    axes = axes.ravel()

    idx_selected = choice(idxes, min([len(axes), len(idxes)]), replace=False)

    for idx_img, ax in zip(idx_selected, axes):
        number_img = data_sample[idx_img, 0, :, :]
        probability = probabilitys[idx_img].max().item()
        predict = predicts[idx_img]
        answer = targets_sample[idx_img].item()

        ax.imshow(number_img)
        ax.set_title(
            f"Predict: {predict}\n"
            f"Answer: {answer}\n"
            f"Prob: {probability * 100.0:.2f}%"
        )
        ax.grid(False)

    return fig, axes


# %% Plot Good result
target_idx = 9
idx_good = where(
    logical_and((predicts == targets_sample), (targets_sample == target_idx))
)
idx_good = idx_good[0].cpu().numpy()
fig, _ = plot_number(idx_good)
fig.suptitle(f"Good Result of {target_idx} images")
fig.tight_layout()

# %% Plot Bad result
idx_bad = where(
    logical_and((predicts != targets_sample), (targets_sample == target_idx))
)
idx_bad = idx_bad[0].cpu().numpy()
fig, _ = plot_number(idx_bad)
fig.suptitle(f"Bad Result of {target_idx} images")
fig.tight_layout()

# %% Plot Confusion matrix
cm = confusion_matrix(predicts, targets_sample, task="multiclass", num_classes=n_class)
cm = cm.cpu().detach().numpy()
fig, ax = plot_confusion_matrix(
    cm, labels, title="Confusion Matrix", cmap_log_scale=True
)

# %% To mps
predicts = predicts.to(device)
data_sample = data_sample.to(device)
targets_sample = targets_sample.to(device)

# %% Plot Filter output
first_layer = Sequential(*list(model.model.children())[:1])
first_layer_output = first_layer(data_sample)[0].detach()

fig, axes = subplots(4, 8, figsize=(10, 6))
for idx, ax in enumerate(axes.flat):
    ax.imshow(first_layer_output[idx].cpu())
    ax.set_title(str(idx))
    ax.set_xticks([])
    ax.set_yticks([])
fig.suptitle("1st Conv2D output")
fig.tight_layout()

# %% Plot Filter output
second_layer = Sequential(*list(model.model.children())[:2])
second_layer_output = second_layer(data_sample)[0].detach()

fig, axes = subplots(8, 8, figsize=(10, 10))
for idx, ax in enumerate(axes.flat):
    ax.imshow(second_layer_output[idx].cpu())
    ax.set_title(str(idx))
    ax.set_xticks([])
    ax.set_yticks([])
fig.suptitle("2nd Conv2D output")
fig.tight_layout()
