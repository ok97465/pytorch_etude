#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Mnist data loader.

Created on Sun Apr 24 20:59:01 2022

@author: ok97465
"""
# %% Import
# Third party imports
import torch as th
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import MNIST
from torchvision.transforms import Compose, ToTensor


class MnistData(LightningDataModule):
    """."""

    def __init__(self, batch_size: int, n_worker: int = 0, path_data: str = "./data"):
        """."""
        super().__init__()
        self.batch_size = batch_size
        self.n_worker = n_worker
        self.path_data = path_data
        self.transform = Compose([ToTensor()])

    def prepare_data(self):
        """Prepare Data."""
        MNIST(root=self.path_data, download=True)

    def setup(self, stage=None):
        """."""
        mnist_all = MNIST(
            root=self.path_data, train=True, transform=self.transform, download=False
        )

        self.train, self.val = random_split(
            mnist_all, [55000, 5000], generator=th.Generator().manual_seed(1)
        )

        self.test = MNIST(
            root=self.path_data, train=False, transform=self.transform, download=False
        )

    def train_dataloader(self) -> DataLoader:
        """."""
        return DataLoader(
            self.train, batch_size=self.batch_size, num_workers=self.n_worker
        )

    def val_dataloader(self) -> DataLoader:
        """."""
        return DataLoader(
            self.val, batch_size=self.batch_size, num_workers=self.n_worker
        )

    def test_dataloader(self) -> DataLoader:
        """."""
        return DataLoader(
            self.test, batch_size=self.batch_size, num_workers=self.n_worker
        )
