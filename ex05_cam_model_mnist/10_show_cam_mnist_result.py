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
from numpy import empty, logical_and, ndarray, newaxis
from numpy.random import choice
from torch import Tensor, argmax, where, zeros
from torch.nn import Module
from torch.nn.functional import interpolate

# Local imports
from dataloader.mnist_dataloader import MnistData
from ex05_cam_model_mnist.cam_model_mnist import MnistCAMModel

# %% Parameters
path_data = "./data"
path_model_ckpt = (
    "./logs/cam_model_mnist/230515_230226_ver0.001/checkpoint/"
    "epoch=49-valid_acc=0.98.ckpt"
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
model = MnistCAMModel().load_from_checkpoint(path_model_ckpt)
model.eval()

model_params = list(model.parameters())
weight_featuremap = model_params[-1]

# %% Prediction
probabilitys = model.forward(data_sample)
predicts = argmax(probabilitys, dim=1)

# %% To CPU
predicts_cpu = predicts.cpu()
data_sample_cpu = data_sample.cpu()
targets_sample_cpu = targets_sample.cpu()

# %% Hook for feature map
class HookModule:
    """Hook for save in/out of module."""

    def __init__(self, module: Module):
        """."""
        self.hook = module.register_forward_hook(self.hook_fn)
        self.d_in: Tensor = empty(0)
        self.d_out: Tensor = empty(0)

    def hook_fn(self, module: Module, d_in: Tensor, d_out: Tensor):
        """."""
        self.d_in = d_in
        self.d_out = d_out


data_of_relu2 = HookModule(model.model._modules.get("relu2"))

# %% Calc CAM
def calc_cam(data: Tensor) -> Tensor:
    """Calc class activiation map.

    Args:
        data: input of model(1 x 1 x h x w).

    Returns:
        ndarray: class activation map.

    """
    n_h, n_w = data.shape[2:]

    logit = model(data)
    idx_ret = logit.argmax()
    weight = weight_featuremap[idx_ret]
    features = data_of_relu2.d_out[0]

    cam = zeros(features.shape[1:], dtype=features.dtype)
    for w, f in zip(weight, features):
        cam += w.cpu() * f.cpu()

    cam = interpolate(
        cam[newaxis][newaxis], size=(n_h, n_w), mode="bilinear", align_corners=False
    )

    return cam[0, 0, :, :]


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

        cam = calc_cam(number_img[newaxis]).cpu().detach().numpy()
        predict = predicts[idx_img].item()
        answer = targets_sample[idx_img].item()

        ax.imshow(cam, cmap="turbo", alpha=0.7)
        ax.contour(number_img[0].cpu(), alpha=1.0)
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
fig.suptitle(f"CAM of Good Result of {target_idx} images")
fig.tight_layout()

# %% Plot Bad result
idx_bad = where(
    logical_and(
        (predicts_cpu != targets_sample_cpu), (targets_sample_cpu == target_idx)
    )
)
idx_bad = idx_bad[0].numpy()
fig, _ = plot_cam(idx_bad)
fig.suptitle(f"CAM of Bad Result of {target_idx} images")
fig.tight_layout()
