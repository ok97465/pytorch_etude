# -*- coding: utf-8 -*-
"""Show the result of Autoencoder.

Created on Sat Apr 16 11:18:07 2022

@author: ok97465
"""
# %% Import
# Third party imports
import torch
from matplotlib.pyplot import subplots
from torch import randn
from torch.autograd.grad_mode import no_grad

# Local imports
from dataloader.mnist_dataloader import MnistData
from ex04_denoise_AE_mnist.denoise_autoencoder_model_mnist import MnistDenoiseAEModel

# %% Parameters
path_data = "./data"
path_model_ckpt = (
    "./logs/denoise_autoencoder_mnist/230514_170812_ver0.001/checkpoint/"
    "epoch=49-valid_loss=0.0014340.ckpt"
)
n_img_sample = 4
noise_std = 0.05

# %% Data Load
device = torch.device("mps" if torch.backends.mps.is_available() else "gpu")
data_loader = MnistData(n_img_sample)
data_loader.setup()
data_sample = next(iter(data_loader.test_dataloader()))[0]
data_sample = data_sample.to(device)

# %% Load Model
model = MnistDenoiseAEModel().load_from_checkpoint(path_model_ckpt)
model.to(device)
model.eval()

# %% forward
with no_grad():
    data_sample += randn(data_sample.shape).to(device) * noise_std  # Add Noise
    data_result = model.forward(data_sample)

data_sample = data_sample.cpu()
data_result = data_result.cpu()
# %% Plot Result
fig, axes = subplots(2, n_img_sample, sharex=True, sharey=True, figsize=(5, 3))

for idx in range(n_img_sample):
    ax_sample = axes[0][idx]
    ax_result = axes[1][idx]

    d_sample = data_sample[idx, 0, :, :]
    d_result = data_result[idx, 0, :, :]

    ax_sample.imshow(d_sample)
    ax_sample.grid(False)
    ax_sample.set_xticks([])
    ax_sample.set_yticks([])

    ax_result.imshow(d_result)
    ax_result.grid(False)
    ax_result.set_xticks([])
    ax_result.set_yticks([])

axes[0][0].set_ylabel("Input")
axes[1][0].set_ylabel("Output")
fig.suptitle("Denoising by Autoencoder")
fig.tight_layout()
