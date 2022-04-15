# -*- coding: utf-8 -*-
"""Show the result of MNIST.

Created on Sat Apr 16 11:18:07 2022

@author: ok97465
"""
# %% Import
# Third party imports
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.pyplot import subplots
from numpy import logical_and, ndarray, unique
from numpy.random import choice
from torch import argmax, where
from torchmetrics.functional import confusion_matrix
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor

# Local imports
from ex02_linear_model_mnist.linear_model_mnist import MnistLinearModel
from helper.helper import plot_confusion_matrix


# %% Parameters
path_data = "./data"
path_model_ckpt = (
    "./logs/linear_model_mnist/220511_202139_ver0.001/checkpoint/"
    "epoch=5-valid_acc=0.94.ckpt"
)

# %% Data Load
data = MNIST(root=path_data, train=False, transform=ToTensor(), download=False)
data.data.requires_grad_(False)
data.targets.requires_grad_(False)
n_targets = len(unique(data.targets))
labels = [str(idx) for idx in range(n_targets)]

# %% Load Model
model = MnistLinearModel.load_from_checkpoint(path_model_ckpt)
model.eval()

# %% Prediction
data_float = data.data.float()
probabilitys = model.forward(data_float)
predicts = argmax(probabilitys, dim=1)


# %% Plot image
def plot_number(
    idxes: ndarray, size: tuple[int, int] = (2, 4)
) -> tuple[Figure, list[Axes]]:
    """Plot number."""
    fig, axes = subplots(*size, sharex=True, sharey=True)
    axes = axes.ravel()

    idx_selected = choice(idxes, len(axes), replace=False)

    for idx_img, ax in zip(idx_selected, axes):
        number_img = data_float[idx_img]
        probability = probabilitys[idx_img].max().item()
        predict = predicts[idx_img]
        answer = data.targets[idx_img].item()

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
idx_good = where(logical_and((predicts == data.targets), (data.targets == target_idx)))
idx_good = idx_good[0].numpy()
fig, _ = plot_number(idx_good)
fig.suptitle(f"Good Result of {target_idx} images")
fig.tight_layout()

# %% Plot Bad result
idx_bad = where(logical_and((predicts != data.targets), (data.targets == target_idx)))
idx_bad = idx_bad[0].numpy()
fig, _ = plot_number(idx_bad)
fig.suptitle(f"Bad Result of {target_idx} images")
fig.tight_layout()

# %% Plot Confusion matrix
cm = confusion_matrix(predicts, data.targets, num_classes=n_targets)
cm = cm.cpu().detach().numpy()
fig, ax = plot_confusion_matrix(
    cm, labels, title="Confusion Matrix", cmap_log_scale=True
)
