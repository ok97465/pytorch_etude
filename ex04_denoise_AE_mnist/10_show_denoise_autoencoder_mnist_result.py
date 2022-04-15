# -*- coding: utf-8 -*-
"""Show the result of Autoencoder.

Created on Sat Apr 16 11:18:07 2022

@author: ok97465
"""
# %% Import
# Third party imports
from jupyterthemes import jtplot
from matplotlib.pyplot import subplots
from torch import randn
from torch.autograd.grad_mode import no_grad

# Local imports
from dataloader.mnist_dataloader import MnistData
from ex04_denoise_AE_mnist.denoise_autoencoder_model_mnist import MnistDenoiseAEModel

jtplot.style("onedork")

# %% Parameters
path_data = "./data"
path_model_ckpt = (
    "./logs/denoise_autoencoder_mnist/220509_204827_ver0.001/checkpoint/"
    "epoch=49-valid_loss=0.0014383.ckpt"
)
n_img_sample = 4
noise_std = 0.05

# %% Data Load
data_loader = MnistData(n_img_sample)
data_loader.setup()
data_sample = next(iter(data_loader.test_dataloader()))[0]

# %% Load Model
model = MnistDenoiseAEModel().load_from_checkpoint(path_model_ckpt)
model.eval()

# %% forward
with no_grad():
    data_sample += randn(data_sample.shape) * noise_std  # Add Noise
    data_result = model.forward(data_sample)

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
