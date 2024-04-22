#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 13 09:10:43 2023

@author: daa
"""
""" 
PyTorch implementation of Linear Context Transform Block

As described in https://arxiv.org/pdf/1909.03834v2

Linear Context Transform (LCT) block divides all channels into different groups
and normalize the globally aggregated context features within each channel group, 
reducing the disturbance from irrelevant channels. Through linear transform of 
the normalized context features, LCT models global context for each channel independently. 
"""



import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

# standardized conv layer
class Conv2d(nn.Conv2d):
    def __init__(self, in_chan, out_chan, kernel_size, stride=1, 
                 padding=0, dilation=1, groups=1, bias=True):
        super(Conv2d, self).__init__(in_chan, out_chan, kernel_size, stride,
                 padding, dilation, groups, bias)

    def forward(self, x):
        weight = self.weight
        weight_mean = weight.mean(dim=1, keepdim=True).mean(dim=2,
                                  keepdim=True).mean(dim=3, keepdim=True)
        weight = weight - weight_mean
        std = weight.view(weight.size(0), -1).std(dim=1).view(-1,1,1,1)+1e-5
        weight = weight / std.expand_as(weight)
    
        return F.conv2d(x, weight, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)


class LCT_original(nn.Module):
    def __init__(self, channels, groups, eps=1e-5):
        super().__init__()
        assert channels % groups == 0, "Number of channels should be evenly divisible by the number of groups"
        self.groups = groups
        self.channels = channels
        self.eps = eps
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.maxpool = nn.AdaptiveMaxPool2d(1)
        self.w = nn.Parameter(torch.ones(channels))
        #self.b = nn.Parameter(torch.zeros(channels))
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        batch_size = x.shape[0]
        y = self.maxpool(x).view(batch_size, self.groups, -1)
        mean = y.mean(dim=-1, keepdim=True)
        mean_x2 = (y ** 2).mean(dim=-1, keepdim=True)
        var = mean_x2 - mean ** 2
        y_norm = (y - mean) / torch.sqrt(var + self.eps)
        y_norm = y_norm.reshape(batch_size, self.channels, 1, 1)
        y_norm = self.w.reshape(1, -1, 1, 1) * y_norm #+ self.b.reshape(1, -1, 1, 1)
        y_norm = self.sigmoid(y_norm)
        return x * y_norm.expand_as(x)
        

class L2NormalizationLayer(nn.Module):
    def __init__(self, dim=1, eps=1e-12):
        super(L2NormalizationLayer, self).__init__()
        self.dim = dim
        self.eps = eps

    def forward(self, x):
        return F.normalize(x, p=2, dim=self.dim, eps=self.eps)
    
class SpatialAttention_normalized(nn.Module):
    def __init__(self, kernel_size, padding, dilation):
    #def __init__(self, kernel_size=7):
        super(SpatialAttention_normalized, self).__init__()
        self.conv = nn.Conv2d(1, 1, kernel_size, padding=padding, dilation=dilation, bias=False)#padding=kernel_size//2
        #self.conv = Conv2d(2, 1, kernel_size, padding=padding, dilation=dilation, bias=False)#padding=kernel_size//2
        
        self.L2 = L2NormalizationLayer()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x1,x2):
        #avg_out = torch.mean(x1, dim=1, keepdim=True)
        
        max_out, _ = torch.max(x1, dim=1, keepdim=True)
        max_out = self.L2(max_out)
        min_out, _ = torch.min(x2, dim=1, keepdim=True)
        min_out = self.L2(min_out)
        #print(max_out.size())
        #out = torch.concat([avg_out, max_out], dim=1)
        out = torch.add(max_out,-min_out)
        #out = torch.concat([out, out], dim=1)

        out = self.conv(out)
        return x2 * self.sigmoid(out) 

class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, w, h = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class LCT(nn.Module):
    def __init__(self, channels, groups, divide_factor, kernel_size, eps=1e-5):
        super().__init__()
        assert channels % groups == 0, "Number of channels should be evenly divisible by the number of groups"
        self.groups = groups
        self.channels = channels
        self.eps = eps
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.maxpool = nn.AdaptiveMaxPool2d(1)#nn.MaxPool2d((int(256/divide_factor),int(256/divide_factor)))
        self.L2 = L2NormalizationLayer()
        #self.alpha = nn.Parameter(torch.ones(1, channels, 1, 1))
        self.w1 = nn.Parameter(torch.ones(channels))
        self.b1 = nn.Parameter(torch.zeros(channels))
        #self.w2 = nn.Parameter(torch.ones(channels))
        #self.b2 = nn.Parameter(torch.zeros(channels))
        
        
        #self.avg_out = torch.mean(channels, dim=1, keepdim=True)
        #self.conv_att = nn.Conv2d(1, 1, kernel_size=3, padding=3//2, dilation=1, bias=False)
        
        #self.s1 = nn.Parameter(torch.ones(channels))
        #self.s2 = nn.Parameter(torch.ones(channels))
        #self.theta = nn.Parameter(torch.ones(channels))
        
        #self.conv_att = nn.Conv2d(1, 1, kernel_size=7, padding=7//2, dilation=1, bias=False)
        #self.conv = nn.Conv2d(channels, channels, kernel_size=1, padding=0, dilation=1, bias=False)                    
        #self.gamma1 = nn.Parameter(torch.zeros(self.groups))
        #self.gamma2 = nn.Parameter(torch.zeros(self.groups))
        #self.gamma_norm = nn.Parameter(torch.zeros(1, channels, 1, 1))
        self.sigmoid = nn.Sigmoid()
        #self.activation = nn.ReLU(inplace=True)
        #self.conv = nn.Conv2d(2*channels, channels, kernel_size, padding=kernel_size//2, bias=False)
        
        #-------------------------------------------------
        #self.sa1 = SpatialAttention_normalized(kernel_size=7, padding=(7)//2,    dilation=1)
        #self.sa2 = SpatialAttention_normalized(kernel_size=7, padding=(7+5)//2,  dilation=2)
        #self.sa3 = SpatialAttention_normalized(kernel_size=7, padding=(7+5+7)//2,dilation=3)
        
        #self.sa_3 = SpatialAttention_normalized(kernel_size=3, padding=3//2, dilation=1)
        #self.sa_7 = SpatialAttention_normalized(kernel_size=7, padding=7//2, dilation=1)
        
        #self.SE = SELayer(channel=channels,reduction=16)
        #self.l2 = L2NormalizationLayer(channels)
        #self.l2p= nn.LayerNorm
        #self.bn1 = nn.GroupNorm(8, channels)
        #out_c=channels
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // 16, bias=False),#reduction=16
            #nn.LayerNorm(channels//16),
            #nn.GroupNorm(channels // 16, channels//16),
            #nn.BatchNorm1d(num_features=channels//16),# Batch normalization
            nn.ReLU(inplace=True),
            #nn.Linear(channels // 16, channels // 16, bias=False),
            #nn.BatchNorm1d(num_features=channels//16),
            #nn.GroupNorm(channels//16, channels//16),
            #nn.ReLU(inplace=True),
            #nn.Linear(channels // 16, channels // 16, bias=False),
            #nn.BatchNorm1d(num_features=channels//16),
            #nn.ReLU(inplace=True),
            #nn.GroupNorm(channels//16, channels//16),
            nn.Linear(channels // 16, channels, bias=False),
            #nn.BatchNorm1d(num_features=channels),
            #nn.LayerNorm(channels),
            #nn.GroupNorm(channels // 16, channels),
            nn.Sigmoid())
        
        #self.compress = nn.Sequential(
        #  nn.Conv2d(in_channels=channels*3, out_channels=int(channels), kernel_size=1, stride=1, padding=0, dilation=1),
          #nn.ReLU(),
        #)
        
    def forward(self, x):
        b, c, w, h = x.size()
        batch_size = x.shape[0]
        ##avg=self.avgpool(x)
        ##print('avg size = ', avg.size())
        #y = self.avgpool(x)#.view(batch_size, self.groups, -1)
        ##print('y size  = ', y.size())
        ##mx=self.maxpool(x)
        ##print('mx size = ', mx.size())
# --- spatial attention--------------------------------------------------------        
        #z=torch.mean(x, dim=1, keepdim=True)
        #z = self.conv_att(z)
        #z = self.sigmoid(z)
        #x = x*z
#------------------------------------------------------------------------------        
        #ymin = torch.min(x, dim=1, keepdim=True).values.view(batch_size, self.groups, -1)
        #min_out = F.normalize(ymin.values, dim=1, p=2)
        #min_out=self.maxpool(-x).view(b,c)#.view(batch_size, self.groups, -1)
        min_out=self.avgpool(x).view(b,c)#view(batch_size, self.groups, -1)
        #min_out=self.maxpool(-x).view(batch_size, self.groups, -1)
        
        ##ymax = torch.max(x, dim=1, keepdim=True).values.view(batch_size, self.groups, -1)
        #min_out = self.gamma1.reshape(1, -1, 1) *min_out#F.normalize(min_out, dim=1, p=2)
        SE_out = self.fc(min_out).view(b,c,1,1)#reshape(1, -1, 1, 1)#.view(batch_size, self.groups, -1)
        #zw=self.fc(min_out).view(b,c,1,1)
        #print('zw size = ', zw.size())
        #print('SE_out size = ', SE_out.size())
        
        y = self.maxpool(x).view(batch_size, self.groups, -1)
        
        mean = y.mean(dim=-1, keepdim=True)
        mean_x2 = (y ** 2).mean(dim=-1, keepdim=True)
        var = mean_x2 - mean ** 2
        y_norm = (y - mean) / torch.sqrt(var + self.eps)
        y_norm = y_norm.reshape(batch_size, self.channels, 1, 1)
        y_norm1 = self.w1.reshape(1, -1, 1, 1) * y_norm + self.b1.reshape(1, -1, 1, 1)#*y_norm**2
        y_norm1=y_norm1.expand_as(x)
        #print('y_norm1 size =', y_norm1.size())
        #max_out=self.maxpool(x).view(b,c)#.view(batch_size, self.groups, -1)
        #max_out=self.maxpool(x).view(batch_size, self.groups, -1)
        #max_out = self.gamma2.reshape(1, -1, 1) *max_out#F.normalize(max_out, dim=1, p=2)
        #max_out = self.fc(max_out).view(batch_size, self.groups, -1)#.view(b, c, 1, 1)
        
        #mean = max_out.mean(dim=-1, keepdim=True)
        #mean_x2 = (max_out ** 2).mean(dim=-1, keepdim=True)
        #var = mean_x2 - mean ** 2
        #y_norm = (max_out - mean) / torch.sqrt(var + self.eps)
        #y_norm = y_norm.reshape(batch_size, self.channels, 1, 1)
        #y_norm2 = y_norm#self.w2.reshape(1, -1, 1, 1) * y_norm + self.b2.reshape(1, -1, 1, 1)#*y_norm**2
        
        #y_norm1 = self.sigmoid(y_norm1)
        #y_norm1=x * y_norm1.expand_as(x)
        #SE_out = self.sigmoid(SE_out)
        #SE_out =x * SE_out.expand_as(x)
        #out_7 = self.sa_7(SE_out,y_norm1)
        
        #out_7 = self.sa_7(x)
        #yout = torch.add(max_out,-min_out)
        
        #yout = torch.add(self.L2(y_norm1)*self.s1.reshape(1, -1, 1, 1), self.L2(SE_out)*self.s2.reshape(1, -1, 1, 1))
        yout = torch.add(self.L2(y_norm1), self.L2(SE_out))
        
        
        
        #yout = torch.add(yout, out_7*self.theta.reshape(1,-1,1,1))
        #yout = torch.concat([yout, yout], dim=1)
        #yout = self.conv(yout)
        #y=yout.reshape(batch_size,self.channels,1,1)
        
        y_norm = self.sigmoid(yout)
        #print('y_norm size =', y_norm.size())
        #y_norm = self.sigmoid(y_norm2)
        #print('yout size =', yout.size())
#-----original LCT -----------------------------------------------------------        
        #mean = y.mean(dim=-1, keepdim=True)
        #mean_x2 = (y ** 2).mean(dim=-1, keepdim=True)
        #var = mean_x2 - mean ** 2
        #y_norm = (y - mean) / torch.sqrt(var + self.eps)
        #y_norm = y_norm.reshape(batch_size, self.channels, 1, 1)
        #y_norm1 = self.w.reshape(1, -1, 1, 1) * y_norm + self.b.reshape(1, -1, 1, 1)
        #y_norm = self.sigmoid(y_norm1)
        #out = x * y_norm.expand_as(x)
        
        out_before = x * y_norm.expand_as(x)
        #out_before = y_norm.expand_as(x)
        #out = y_norm
        
        #out1 = self.activation(self.sa1(out_before))#ks=7
        #out2 = self.activation(self.sa2(out_before))
        #out3 = self.activation(self.sa3(out_before))#ks=3
        
        #out_3 = self.sa_3(out_before)
        
        #out_7 = self.sa_7(out_before)
        
        #out = self.conv(torch.cat((out1,out3),dim=1))# very slow and the results are not great
        
        #av1=self.fc(self.avgpool(out_3).view(b,c))
        #av2=self.fc(self.avgpool(out_7).view(b,c))
        #av1=av1.reshape(batch_size, self.channels, 1, 1)
        #av2=av2.reshape(batch_size, self.channels, 1, 1)
        #av1=av1.expand_as(x)
        #av2=av2.expand_as(x)
        #out = torch.max(out_3*av1,out_7*av2)
        
        #out_before = out_before*out_7#torch.max(out_3, out_7)
        #out_before = x*out_7
        #out = torch.add(out1,out2)
        #out = torch.add(out,out3)
        #out = torch.div(out,3)
        #out = torch.min(out,out2)
        #out = torch.add(out,out3)
        #out = torch.div(out,2)
        #out = self.compress(torch.cat((out1,out2,out3),dim=1))
        #out = torch.cat((self.compress(out1),self.compress(out3)), dim=1)
        #out = self.sa1(out)
        return out_before#torch.tanh(LCT_norm)
        

#if __name__ == "__main__":
#    x = torch.randn(2, 64, 32, 32)
#    attn = LCT(64, 8)
#    y = attn(x)
#    print(y.shape)