#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 28 12:32:14 2023

@author: https://github.com/changzy00/pytorch-attention
"""
#Gaussian Context Transformer
import torch
from torch import nn

class GaussianCT(nn.Module):
    def __init__(self, channels, c=2, eps=1e-5):
        super().__init__()
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.eps = eps
        self.c = c

    def forward(self, x):
        y = self.avgpool(x)
        mean = y.mean(dim=1, keepdim=True)
        mean_x2 = (y ** 2).mean(dim=1, keepdim=True)
        var = mean_x2 - mean ** 2
        y_norm = (y - mean) / torch.sqrt(var + self.eps)
        y_transform = torch.exp(-(y_norm ** 2 / 2 * self.c))
        return x * y_transform.expand_as(x)
    
#if __name__ == "__main__":
#    x = torch.randn(2, 64, 32, 32)
#    attn = GCT(64)
#    y = attn(x)
#    print(y.shape)
