#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""GradCAM.

Ref : https://github.com/1Konny/gradcam_plus_plus-pytorch

Created on Sun May  8 09:45:40 2022

@author: ok97465
"""
# Third party imports
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Module


class GradCAM(object):
    """Calculate GradCAM salinecy map.

    A simple example:

    Args:
        model: model
        target_layer : target layer for gradcam
    """

    def __init__(self, model: Module, target_layer: Module):
        """."""
        self.model_arch = model

        self.gradients = dict()
        self.activations = dict()

        target_layer.register_forward_hook(self.forward_hook)
        target_layer.register_full_backward_hook(self.backward_hook)

    def backward_hook(self, module, grad_input, grad_output):
        """."""
        self.gradients["value"] = grad_output[0]

    def forward_hook(self, module, input, output):
        """."""
        self.activations["value"] = output

    def forward(
        self, input, class_idx=None, retain_graph=False
    ) -> tuple[Tensor, Tensor]:
        """Forward.

        Args:
            input: input image with shape of (1, 1, H, W)
            class_idx (int): class index for calculating GradCAM.
                    If not specified, the class index that makes the highest model
                    prediction score will be used.
            retain_graph: DESCRIPTION. Defaults to False.

        Returns:
            mask: saliency map of the same spatial dimension with input
            logit: model output

        """
        b, c, h, w = input.size()

        logit = self.model_arch(input)
        if class_idx is None:
            score = logit[:, logit.max(1)[-1]].squeeze()
        else:
            score = logit[:, class_idx].squeeze()

        self.model_arch.zero_grad()
        score.backward(retain_graph=retain_graph)
        gradients = self.gradients["value"]
        activations = self.activations["value"]
        b, k, u, v = gradients.size()

        alpha = gradients.view(b, k, -1).mean(2)
        weights = alpha.view(b, k, 1, 1)

        saliency_map = (weights * activations).sum(1, keepdim=True)
        saliency_map = F.relu(saliency_map)
        saliency_map = F.interpolate(
            saliency_map, size=(h, w), mode="bilinear", align_corners=False
        )
        saliency_map_min, saliency_map_max = saliency_map.min(), saliency_map.max()
        saliency_map = (
            (saliency_map - saliency_map_min)
            .div(saliency_map_max - saliency_map_min)
            .data
        )

        return saliency_map, logit

    def __call__(self, input, class_idx=None, retain_graph=False):
        """."""
        return self.forward(input, class_idx, retain_graph)
