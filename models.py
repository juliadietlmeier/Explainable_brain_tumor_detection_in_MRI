#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 24 08:45:46 2024

@author: <julia.dietlmeier@insight-centre.org>
"""

import torch
import torch.nn as nn
from torchvision import models
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable
from GaussianCT import GaussianCT

device="cuda"

"=== ImageNet ================================================================="
resnet50imagenet = models.resnet50(pretrained=True) 
mod = nn.Sequential(*list(resnet50imagenet.children())[:-2])
mod1 = nn.Sequential(*list(resnet50imagenet.children())[:-3])# last stage from resnet50 removed

"=== RadImageNet =============================================================="
resnet50_R = models.resnet50(pretrained=False)
resnet50_radimagenet_weights = torch.load('/home/daa/Desktop/Grace_ML_Labs/Code/RadImageNet_pytorch/ResNet50.pt', map_location=torch.device(device))
resnet50_R = nn.Sequential(*list(resnet50_R.children())[:9])

state_dict = torch.load('/home/daa/Desktop/Grace_ML_Labs/Code/RadImageNet_pytorch/ResNet50.pt', map_location=torch.device(device))
new_state_dict = {}
for k, v in state_dict.items():
            new_state_dict[k[9:]] = v

resnet50_R.load_state_dict(new_state_dict)
mod_R = nn.Sequential(*list(resnet50_R.children())[:-2])
"=============================================================================="

class classifier1(nn.Module):
    def __init__(self):        
        super(classifier1,self).__init__() 
        
        self.fc=nn.Linear(2048,3)
        torch.nn.init.xavier_uniform(self.fc.weight)

    def forward(self,x):     
        
        x = x.mean([2, 3])
        x=self.fc(x)
        
        return  F.softmax(x,dim=1)

class classifier2(nn.Module):
    def __init__(self):
        super(classifier2,self).__init__() 
       
        self.fc=nn.Linear(1024,3)
               
    def forward(self,x):     

        x = x.mean([2, 3])
        x = self.fc(x)
        output = F.softmax(x,dim=1)        
        
        return  output

class classifier3(nn.Module):
    def __init__(self):
        super(classifier3,self).__init__() 
       
        self.fc=nn.Linear(512,3)

    def forward(self,x):     

        x = x.mean([2, 3])
        x = self.fc(x)
        
        return  F.softmax(x,dim=1)

def resnet50only():
    return nn.Sequential(mod,classifier1())

# SEA3: three squeeze and expand blocks ---------------------------------------
class ConvHead(nn.Module):
    def __init__(self):
        super(ConvHead,self).__init__() 
#-- Block 1 -------------------------------------------------------------------
        self.myconv1 = nn.Conv2d(1024,512, kernel_size=(1, 1), padding=0, stride=(1, 1), bias=True)
        torch.nn.init.xavier_uniform(self.myconv1.weight)
        self.relu1 = nn.ReLU(inplace=True)
        self.att1 = GaussianCT(512,1)
        
        self.myconv2 = nn.Conv2d(512,1024, kernel_size=(3, 3), padding=1, stride=(1, 1), bias=True)
        torch.nn.init.xavier_uniform(self.myconv2.weight)
        self.relu2 = nn.ReLU(inplace=True)
        self.att2 = GaussianCT(1024,1)
#-- Block 2 -------------------------------------------------------------------        
        self.myconv3 = nn.Conv2d(1024,512, kernel_size=(1, 1), padding=0, stride=(1, 1), bias=True)
        torch.nn.init.xavier_uniform(self.myconv3.weight)
        self.relu3 = nn.ReLU(inplace=True)
        self.att3 = GaussianCT(512,1)
        
        self.myconv4 = nn.Conv2d(512,1024, kernel_size=(3, 3), padding=1, stride=(1, 1), bias=True)
        torch.nn.init.xavier_uniform(self.myconv4.weight)
        self.relu4 = nn.ReLU(inplace=True)
        self.att4 = GaussianCT(1024,1)
#-- Block 3 -------------------------------------------------------------------        
        self.myconv5 = nn.Conv2d(1024,512, kernel_size=(1, 1), padding=0, stride=(1, 1), bias=True)
        torch.nn.init.xavier_uniform(self.myconv5.weight)
        self.relu5 = nn.ReLU(inplace=True)
        self.att5 = GaussianCT(512,1) 
        
        self.myconv6 = nn.Conv2d(512,1024, kernel_size=(3, 3), padding=1, stride=(1, 1), bias=True)
        torch.nn.init.xavier_uniform(self.myconv6.weight)
        self.relu6 = nn.ReLU(inplace=True)
        self.att6 = GaussianCT(1024,1)
        
    def forward(self,x):     

        x = self.myconv1(x)
        x = self.relu1(x)
        x = self.att1(x)
        
        x = self.myconv2(x)
        x = self.relu2(x)
        x = self.att2(x)

#------------------------------------------------------------------------------        
        x = self.myconv3(x)
        x = self.relu3(x)
        x = self.att3(x)
        
        x = self.myconv4(x)
        x = self.relu4(x)
        x = self.att4(x)

#------------------------------------------------------------------------------        
        x = self.myconv5(x)
        x = self.relu5(x)
        x = self.att5(x)
        
        x = self.myconv6(x)
        x = self.relu6(x)
        x = self.att6(x)
        
        return  x

class ConvHead2(nn.Module):
    def __init__(self):
        super(ConvHead2,self).__init__() 
#-- Block 1 -------------------------------------------------------------------
        self.myconv1 = nn.Conv2d(1024,512, kernel_size=(1, 1), padding=0, stride=(1, 1), bias=True)
        torch.nn.init.xavier_uniform(self.myconv1.weight)
        self.relu1 = nn.ReLU(inplace=True)
        self.att1 = GaussianCT(512,1)
        
        self.myconv2 = nn.Conv2d(512,1024, kernel_size=(3, 3), padding=1, stride=(1, 1), bias=True)
        torch.nn.init.xavier_uniform(self.myconv2.weight)
        self.relu2 = nn.ReLU(inplace=True)
        self.att2 = GaussianCT(1024,1)
#-- Block 2 -------------------------------------------------------------------        
        self.myconv3 = nn.Conv2d(1024,512, kernel_size=(1, 1), padding=0, stride=(1, 1), bias=True)
        torch.nn.init.xavier_uniform(self.myconv3.weight)
        self.relu3 = nn.ReLU(inplace=True)
        self.att3 = GaussianCT(512,1)
        
        self.myconv4 = nn.Conv2d(512,1024, kernel_size=(3, 3), padding=1, stride=(1, 1), bias=True)
        torch.nn.init.xavier_uniform(self.myconv4.weight)
        self.relu4 = nn.ReLU(inplace=True)
        self.att4 = GaussianCT(1024,1)
        
    def forward(self,x):     

        x = self.myconv1(x)
        x = self.relu1(x)
        x = self.att1(x)
        
        x = self.myconv2(x)
        x = self.relu2(x)
        x = self.att2(x)

#------------------------------------------------------------------------------        
        x = self.myconv3(x)
        x = self.relu3(x)
        x = self.att3(x)
        
        x = self.myconv4(x)
        x = self.relu4(x)
        x = self.att4(x)

        return  x    

class ConvHead3(nn.Module):
    def __init__(self):
        super(ConvHead3,self).__init__() 
#-- Block 1 -------------------------------------------------------------------
        self.myconv1 = nn.Conv2d(1024,512, kernel_size=(1, 1), padding=0, stride=(1, 1), bias=True)
        torch.nn.init.xavier_uniform(self.myconv1.weight)
        self.relu1 = nn.ReLU(inplace=True)
        self.att1 = GaussianCT(512,1)
        
        self.myconv2 = nn.Conv2d(512,1024, kernel_size=(3, 3), padding=1, stride=(1, 1), bias=True)
        torch.nn.init.xavier_uniform(self.myconv2.weight)
        self.relu2 = nn.ReLU(inplace=True)
        self.att2 = GaussianCT(1024,1)
        
    def forward(self,x):     

        x = self.myconv1(x)
        x = self.relu1(x)
        x = self.att1(x)
        
        x = self.myconv2(x)
        x = self.relu2(x)
        x = self.att2(x)

        return  x   
    
class ConvHead4(nn.Module):
    def __init__(self):
        super(ConvHead4,self).__init__() 
#-- Block 1 -------------------------------------------------------------------
        self.myconv1 = nn.Conv2d(1024,512, kernel_size=(1, 1), padding=0, stride=(1, 1), bias=True)
        torch.nn.init.xavier_uniform(self.myconv1.weight)
        self.relu1 = nn.ReLU(inplace=True)
        self.att1 = GaussianCT(512,1)
        
    def forward(self,x):     

        x = self.myconv1(x)
        x = self.relu1(x)
        x = self.att1(x)

        return  x      

class B2_Net(nn.Module):
    def __init__(self):
        super(B2_Net, self).__init__()
        
        
        self.myconv1 = nn.Conv2d(3, 16, kernel_size=(5,5))
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=2)
        
        self.myconv2 = nn.Conv2d(16, 32, kernel_size=(4,4))
        self.myconv3 = nn.Conv2d(32, 64, kernel_size=(3,3))
        self.myfc1 = nn.Linear(40000, 16)
        self.myfc2 = nn.Linear(16, 3)
        
    def forward(self, x):
        
        x = self.myconv1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.myconv2(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.myconv3(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = torch.flatten(x, 1)
        x = self.myfc1(x)
        x = self.relu(x)
        x = self.myfc2(x)
        
        output = F.softmax(x,dim=1) 
        return output
    
def SEA3_resnet50_RadImageNet():
    mod11 = nn.Sequential(mod_R,ConvHead())
    return nn.Sequential(mod11,classifier2())

def SEA3_resnet50():
    mod11 = nn.Sequential(mod1,ConvHead())
    return nn.Sequential(mod11,classifier2())

def SEA2_resnet50():
    mod11 = nn.Sequential(mod1,ConvHead2())
    return nn.Sequential(mod11,classifier2())

def SEA1_resnet50():
    mod11 = nn.Sequential(mod1,ConvHead3())
    return nn.Sequential(mod11,classifier2())

def SEA0_resnet50():
    mod11 = nn.Sequential(mod1,ConvHead4())
    return nn.Sequential(mod11,classifier3())



