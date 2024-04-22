#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 21 13:10:42 2024

@author: daa
"""
from BaseCAM import BaseCAM
from svd_on_activations import get_2d_projection

# https://arxiv.org/abs/2008.00299


class EigenCAM(BaseCAM):
    def __init__(self, model, target_layers, 
                 reshape_transform=None):
        super(EigenCAM, self).__init__(model,
                                       target_layers,
                                       reshape_transform,
                                       uses_gradients=False)

    def get_cam_image(self,
                      input_tensor,
                      target_layer,
                      target_category,
                      activations,
                      grads,
                      eigen_smooth):
        return get_2d_projection(activations)