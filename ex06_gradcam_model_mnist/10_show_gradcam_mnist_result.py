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
from numpy.random import choice
from numpy import logical_and, ndarray, newaxis
from torch import where

# Local imports
from dataloader.mnist_dataloader import MnistData
from ex03_cnn_model_mnist.cnn_model_mnist import MnistCnnModel
from ex06_gradcam_model_mnist.gradcam import GradCAM

# %% Parameters
path_data = "./data"
path_model_ckpt = (
    "./logs/cnn_model_mnist/230513_093434_ver0.001/checkpoint/"
    "epoch=24-valid_acc=0.99.ckpt"
)  # model trained in ex03
n_img_sample = 10000

# %% Data Load
device = torch.device("mps" if torch.backends.mps.is_available() else "gpu")
data_loader = MnistData(n_img_sample)
data_loader.setup()
data_sample, targets_sample = next(iter(data_loader.test_dataloader()))
data_sample = data_sample.to(device)
targets_sample = targets_sample.to(device)

# %% Load Model
model = MnistCnnModel().load_from_checkpoint(path_model_ckpt)
model.eval()
target_layer = model.model._modules.get("conv_block2")._modules.get("conv2d")
calc_cam = GradCAM(model, target_layer)

# %% Prediction
probabilitys = model.forward(data_sample)
predicts = probabilitys.argmax(dim=1)

# %% To CPU
predicts_cpu = predicts.cpu()
data_sample_cpu = data_sample.cpu()
targets_sample_cpu = targets_sample.cpu()


# %% Plot cam
def plot_cam(
    idxes: ndarray, size: tuple[int, int] = (2, 4)
) -> tuple[Figure, list[Axes]]:
    """Plot number."""
    fig, axes = subplots(*size, sharex=True, sharey=True)
    axes = axes.ravel()

    idx_selected = choice(idxes, min([len(axes), len(idxes)]), replace=False)

    for idx_img, ax in zip(idx_selected, axes):
        number_img = data_sample[idx_img]

        cam, _ = calc_cam(number_img[newaxis])
        cam = cam[0, 0, :, :].cpu().detach().numpy()
        predict = predicts[idx_img].item()
        answer = targets_sample[idx_img].item()

        ax.imshow(cam, cmap="turbo", alpha=0.5)
        ax.contour(number_img[0].cpu(), alpha=0.8)
        ax.set_title(f"Predict: {predict}\n" f"Answer: {answer}")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(False)

    return fig, axes


# %% Plot Good result
target_idx = 1
idx_good = where(
    logical_and(
        (predicts_cpu == targets_sample_cpu), (targets_sample_cpu == target_idx)
    )
)
idx_good = idx_good[0].numpy()
fig, _ = plot_cam(idx_good)
fig.suptitle(f"GradCAM of Good Result of {target_idx} images")
fig.tight_layout()

# %% Plot Bad result
idx_bad = where(
    logical_and(
        (predicts_cpu != targets_sample_cpu), (targets_sample_cpu == target_idx)
    )
)
idx_bad = idx_bad[0].numpy()
fig, _ = plot_cam(idx_bad)
fig.suptitle(f"GradCAM of Bad Result of {target_idx} images")
fig.tight_layout()
