#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 20 07:32:18 2024

@author: <julia.dietlmeier@insight-centre.org>
"""
import cv2
import torch
import torch.nn as nn
from torchvision import models
import torch.nn.functional as F
import numpy as np
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.autograd import Variable
from PIL import Image
from PIL import ImageFile
import glob
import random
import torch.nn.utils.prune as prune
from tqdm.auto import tqdm
from eigenCAM.EigenCAM import EigenCAM

ImageFile.LOAD_TRUNCATED_IMAGES = True


device="cuda"

seed = 42
if device == "cuda":
    torch.cuda.manual_seed_all(seed)  # GPU seed
    torch.backends.cudnn.deterministic = True  # fix the GPU to deterministic mode
    #torch.backends.cudnn.benchmark = False# this was not set for the comparisons 
    torch.manual_seed(seed)  # CPU seed
    random.seed(seed)  # python seed for image transformation
    np.random.seed(seed)

def save_model(epochs, model, optimizer, criterion):
    """
    Function to save the trained model to disk.
    """
    print(f"Saving final model...")
    torch.save({
                'epoch': epochs,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': criterion,
                }, '/home/daa/Desktop/Grace_ML_Labs/Code/final_model.pt')

import torch
import torch.nn as nn
import numpy as np
#import FLOP

PRUNE_ID = 0
DEBUG_MODE = False

## This code prunes all the Conv2d layers in a given pytorch model. The Conv2d are pruned by removing
## channels based on an evaluation of their weights. The pruning is done with these restrictions:
## 1. Each Conv2d after pruning will retain at least 1 channel
## 2. Conv2d layers with groups != 1 or bias != False are not pruned
## After pruning, a zero_padding layer is added to pad the output tensor up to the correct dimensions

## To use the pruning, write something like model = prune_model(model, factor_removed=)
## args: model - your pytorch model
##       factor_removed - the proportion of layers pruning will try to remove

## Idea is from 'Pruning Filters for Efficient ConvNets' by Hao Li, et al
## (https://arvix.org/abs/1608.08710)

class L2NormalizationLayer(nn.Module):
    def __init__(self, dim=1, eps=1e-12):
        super(L2NormalizationLayer, self).__init__()
        self.dim = dim
        self.eps = eps

    def forward(self, x):
        return F.normalize(x, p=2, dim=self.dim, eps=self.eps)

class debug_mode(object):
    def __enter__(self):
    	global DEBUG_MODE
    	self.prev = DEBUG_MODE
    	DEBUG_MODE = True
    def __exit__(self, *args):
        DEBUG_MODE = self.prev

def mask(model, cut_off=0):
	for p in model.parameters():
		p_mask = abs(p)>cut_off
		p *= p_mask.float()
	return model

def layer_eval(layer):
	element_squared = [e.item()**2 for e in layer.view(-1)]
	return sum(element_squared)

def unwrap_model(model):
	# loops through all layers of model, including inside modulelists and sequentials
	layers = []
	def unwrap_inside(modules):
		for m in modules.children():
			if isinstance(m,nn.Sequential):
				unwrap_inside(m)
			elif isinstance(m,nn.ModuleList):
				for m2 in m:
					unwrap_inside(m2)
			else:
				layers.append(m)
	unwrap_inside(model)
	return nn.ModuleList(layers)

class zero_padding(nn.Module):
	#my version of zero padding, pads up to given number of channels, at the specified index
	def __init__(self, num_channels, keep_channel_idx):
		super(zero_padding, self).__init__()
		self.num_channels = num_channels
		self.keep_channel_idx = keep_channel_idx
	def forward(self,x):
		output = torch.zeros(x.size()[0],self.num_channels,x.size()[2],x.size()[3])
		output[:,self.keep_channel_idx,:,:] = x
		return output

class pruned_conv2d(nn.Module):
	def __init__(self, conv2d, cut_off=0.0):
		super(pruned_conv2d, self).__init__()
		self.in_channels = conv2d.in_channels
		self.out_channels = conv2d.out_channels
		self.kernel_size = conv2d.kernel_size
		self.stride = conv2d.stride
		self.padding = conv2d.padding
		self.dilation = conv2d.dilation
		self.groups = conv2d.groups
		self.bias = conv2d.bias
		global PRUNE_ID
		self.id = PRUNE_ID
		PRUNE_ID+=1
		self.keep_channel = []
		self.keep_channel_idx = []

		if self.groups != 1 or self.bias != None:
			self.new_conv2d = conv2d
		else:
			for idx, channel in enumerate(conv2d.weight):
				if layer_eval(channel)>cut_off:
					self.keep_channel.append(torch.unsqueeze(channel,0))
					self.keep_channel_idx.append(idx)
			if len(self.keep_channel_idx) == 0:
				# if no channels are above cut-off, keep the best channel
				best_channel_eval = 0
				for idx, channel in enumerate(conv2d.weight):
					if layer_eval(channel) > best_channel_eval:
						best_channel = channel
						best_channel_idx = idx
				self.keep_channel.append(torch.unsqueeze(best_channel,0))
				self.keep_channel_idx.append(best_channel_idx)
			self.new_conv2d = nn.Conv2d(in_channels=self.in_channels,
										out_channels=len(self.keep_channel_idx),
										kernel_size=self.kernel_size,
										stride=self.stride,
										padding=self.padding,
										dilation=self.dilation,
										bias=False)
			self.new_conv2d.weight = torch.nn.Parameter(torch.cat(self.keep_channel,0))
			self.zero_padding = zero_padding(self.out_channels, self.keep_channel_idx)

	def forward(self,x):
		if self.groups != 1 or self.bias != None:
			return self.new_conv2d(x)
		else:
			if DEBUG_MODE:
				try:
					x = self.new_conv2d(x)
				except Exception as e:
					print('failed here')
					print('input size: '+ str(x.size()))
					print('layer: ' + str(self.new_conv2d))
					print('layer weight: ' +str(self.new_conv2d.weight.size()))
					print(str(e))
					quit()
			else:
				x = self.new_conv2d(x)
			return self.zero_padding(x)
class SaveBestModel:
    """
    Class to save the best model while training. If the current epoch's 
    validation loss is less than the previous least less, then save the
    model state.
    """
    def __init__(
        self, best_valid_loss=float('inf')
    ):
        self.best_valid_loss = best_valid_loss
        
    def __call__(
        self, current_valid_loss, 
        epoch, model, optimizer, criterion
    ):
        if current_valid_loss < self.best_valid_loss:
            self.best_valid_loss = current_valid_loss
            print(f"\nBest validation loss: {self.best_valid_loss}")
            print(f"\nSaving best model for epoch: {epoch+1}\n")
            torch.save({
                'epoch': epoch+1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': criterion,
                }, '/home/daa/Desktop/Grace_ML_Labs/Code/best_model.pt')
def save_model(epochs, model, optimizer, criterion):
    """
    Function to save the trained model to disk.
    """
    print(f"Saving final model...")
    torch.save({
                'epoch': epochs,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': criterion,
                }, '/home/daa/Desktop/Grace_ML_Labs/Code/final_model.pt')
save_best_model = SaveBestModel()
"=============================================================================="
batch_size=16
num_epochs=100

normalize = transforms.Normalize(
   mean=[0.485, 0.456, 0.406],
   std=[0.229, 0.224, 0.225]
)

normalize_RIN = transforms.Normalize(
   mean=[127.5, 127.5, 127.5],
   std=[127.7, 127.5, 127.5]
)

img_transform = transforms.Compose([transforms.Resize((224, 224)),transforms.ToTensor(),
                                    normalize])

train_data_dir = datasets.ImageFolder("/home/daa/Desktop/Grace_ML_Labs/cheng_dataset_train/", transform = img_transform)

train_loader = torch.utils.data.DataLoader(train_data_dir, batch_size=batch_size, shuffle=True)

#---- on ImageNet -------------------------------------------------------------
#resnet50 = models.resnet50(pretrained=True) 
#---- on RadImageNet ----------------------------------------------------------
resnet50 = models.resnet50(pretrained=False)
resnet50_radimagenet_weights = torch.load('/home/daa/Desktop/Grace_ML_Labs/Code/RadImageNet_pytorch/ResNet50.pt', map_location=torch.device(device))
resnet50 = nn.Sequential(*list(resnet50.children())[:9])


state_dict = torch.load('/home/daa/Desktop/Grace_ML_Labs/Code/RadImageNet_pytorch/ResNet50.pt', map_location=torch.device(device))
new_state_dict = {}
for k, v in state_dict.items():
            new_state_dict[k[9:]] = v

#resnet50.load_state_dict(resnet50_radimagenet_weights)#, strict=False)
resnet50.load_state_dict(new_state_dict)#, strict=False)

#------------------------------------------------------------------------------
#mod = nn.Sequential(*list(resnet50.children())[:-2])# remove fully connected layer and GlobalAvgPool
mod = nn.Sequential(*list(resnet50.children())[:-2])# for RadImageNet. Remove fully connected layer and GlobalAvgPool and last conv block

class convNet(nn.Module):
    def __init__(self):
        super(convNet,self).__init__() 
        #img = images
        self.myconv1 = nn.Conv2d(1024,2048, kernel_size=(1, 1), stride=(1, 1), bias=True)
        torch.nn.init.xavier_uniform(self.myconv1.weight)
        #self.dropout = nn.Dropout(0.1)
        #self.l2  = L2NormalizationLayer()
        #self.bn1 = nn.BatchNorm2d(512)
        #self.myconv2 = nn.Conv2d(512,256, kernel_size=(1, 1), stride=(1, 1), bias=True)
        #torch.nn.init.xavier_uniform(self.myconv2.weight)
        #self.bn2 = nn.BatchNorm2d(256)
        self.relu = nn.ReLU()
        
    def forward(self,x):     
        #print('x size =',x.size())
        #x=x.view(512,7*7).mean(1).view(1,-1)# this does not work
        x = self.myconv1(x)
        #x = self.dropout(x)
        #x = self.bn1(x)
        #x = self.myconv2(x)
        #x = self.bn2(x)
        x = self.relu(x)
        return  x

class Net(nn.Module):
    def __init__(self):
        super(Net,self).__init__() 
        #img = images
        #self.myconv = nn.Conv2d(1024,2048,kernel_size=(2, 2), stride=(1, 1), bias=False)
        #torch.nn.init.xavier_uniform(self.myconv.weight)
        #self.dropout = nn.Dropout(0.5)
        #self.l2  = L2NormalizationLayer()
        
        self.fc=nn.Linear(2048,3)
        #self.fc.weight.data.fill_(1.0)
        torch.nn.init.xavier_uniform(self.fc.weight)
        #self.fc.bias.data=torch.ones(3,512)
        
    def forward(self,x):     
        #print('x size =',x.size())
        #x=x.view(512,7*7).mean(1).view(1,-1)# this does not work
        #x = self.myconv(x)
        #x = self.dropout(x)
        #x = self.l2(x)
        
        x = x.mean([2, 3])# average pooling
        x=self.fc(x)
        return  F.softmax(x,dim=1)
 
mod = nn.Sequential(mod,convNet())
model=nn.Sequential(mod,Net()).cuda()
#---- freeze-------------------------------------------------------------------
trainable_parameters = []
for name, p in model.named_parameters():
    if "fc" or "myconv1" or "myconv2" in name:
        trainable_parameters.append(p)
#------------------------------------------------------------------------------
class_weights = [1.4461315979754157, 0.7183908045977011, 1.0911074740861975]
class_weights = torch.FloatTensor(class_weights).cuda()

#optimizer = torch.optim.SGD(params=trainable_parameters, lr=0.001, momentum=1e-5)
optimizer = torch.optim.AdamW(params=trainable_parameters, lr=0.00001)  
#criterion = nn.CrossEntropyLoss(weight=class_weights)
criterion = nn.CrossEntropyLoss()

"train========================================================================="

total_step = len(train_loader)
loss_list = []
acc_list = []
loss_arr=[]

for epoch in range(num_epochs):
    #for i, (images, labels) in enumerate(train_loader):
    for i, data in tqdm(enumerate(train_loader),total=len(train_loader)):
        images,labels = data
        images=images.cuda()
        labels=labels.cuda()
        # Run the forward pass
        outputs = model(images)
        #print(outputs.shape)
        loss = criterion(outputs, labels)
        loss_list.append(loss.item())

        # Backprop 
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Track the accuracy
        total = labels.size(0)
        _, predicted = torch.max(outputs.data, 1)
        correct = (predicted == labels).sum().item()
        acc_list.append(correct / total)

        if (i + 1) % 100 == 0:
            print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Accuracy: {:.2f}%'
                  .format(epoch + 1, num_epochs, i + 1, total_step, loss.item(),
                          (correct / total) * 100))
            loss_arr.append(loss.item())
    save_best_model(loss, epoch, model, optimizer, criterion)
            
        #save_model(epoch, model, optimizer, criterion)
        
torch.save(model.state_dict(), 'resnet50_100epochs_RadImageNet.pt')
"=============================================================================="
            
def return_CAM(feature_conv, weight, class_idx):
    # generate the class -activation maps upsample to 256x256
    size_upsample = (256, 256)
    bz, nc, h, w = feature_conv.shape
    output_cam = []
    for idx in class_idx:
        beforeDot =  feature_conv.reshape((nc, h*w))
        
        print('beforeDot size = ',np.shape(beforeDot))
        cam = np.matmul(weight[idx], beforeDot)
        cam = cam.reshape(h, w)
        cam = cam - np.min(cam)
        cam_img = cam / np.max(cam)
        cam_img = np.uint8(255 * cam_img)
        output_cam.append(cv2.resize(cam_img, size_upsample))
    return output_cam

class MyLayer(nn.Module):
    """ Custom Linear layer but mimics a standard linear layer """
    def __init__(self,weight_in):
        super().__init__()

        self.weights = weight_in  # nn.Parameter is a Tensor that's a module parameter.
        bias = torch.Tensor(3,2048)
        self.bias = nn.Parameter(bias)
        
    def forward(self, x):
        w_times_x= torch.mm(x, self.weights.t())
        return self.weights.t()#torch.add(w_times_x, self.bias)  # w times x + b

def return_pruned_CAM(params, feature_conv, weights, class_idx, norm, PR, seed=42):
    # generate the class -activation maps upsample to 256x256
    size_upsample = (256, 256)
    bz, nc, h, width = feature_conv.shape
    output_cam = []
    for idx in class_idx:
        beforeDot =  feature_conv.reshape((nc, h*width))
        
        #weight = torch.squeeze(params[-2])
        
        mydummyconv=nn.Conv2d(2048,1,kernel_size=(1,1))
        #conv=MyLayer(weight[idx])
        mydummyconv.weight.data=torch.from_numpy(weights)#[idx]
        mydummyconv.bias.data=torch.zeros(3,2048)
        
        print('weights.size=',np.shape(weights))
        
        #fc_parameters=[]
        #for name, p in model.named_parameters():
        #    if "fc" in name:
        #        fc_parameters.append(p)
        #parameters_to_prune = ((mydummyconv, "weight"))         
        #prune.random_unstructured(mydummyconv, name='weight', amount=PR)#, name="weight"
#------------------------------------------------------------------------------        
        prune.ln_structured(mydummyconv, name='weight',amount=PR, n=norm, dim=1)# this is working
        
    
        #parameters_to_prune = [
        #    (module, "weight") for module in filter(lambda m: type(m) == mydummyconv, model.modules())]
        #prune.global_unstructured(
        #    parameters_to_prune,
        #    pruning_method=prune.L1Unstructured,
        #    amount=0.2,
        #    )
    
        w=mydummyconv.weight.detach().numpy()
        print(np.shape(w))
        
        #w=np.squeeze(w,-1)
        #w=np.squeeze(w,-1)
        #print(np.shape(w))
        cam = np.matmul(w, beforeDot)
        print(np.shape(cam))
        cam = np.array(cam[idx]).reshape(h, width)
        
        cam = cam - np.min(cam)
        cam_img = 1-(cam / np.max(cam))
        cam_img = np.uint8(255 * cam_img)
        output_cam.append(cv2.resize(cam_img, size_upsample))
    return output_cam



normalize = transforms.Normalize(
   mean=[0.485, 0.456, 0.406],
   std=[0.229, 0.224, 0.225]
)

normalize_RIN = transforms.Normalize(
   mean=[127.5, 127.5, 127.5],
   std=[127.7, 127.5, 127.5]
)

preprocess = transforms.Compose([
   transforms.Resize((224,224)),
   transforms.ToTensor(),
   normalize
])


"test=========================================================================="

state_dict = torch.load('/home/daa/Desktop/Grace_ML_Labs/Code/resnet50_100epochs_RadImageNet.pt', map_location=torch.device(device))
model.load_state_dict(state_dict)
#state_dict = torch.load('/home/daa/Desktop/Grace_ML_Labs/Code/best_model.pt', map_location=torch.device(device))
#model.load_state_dict(state_dict['model_state_dict'])

#params = list(Net().parameters())

params = list(model.parameters())
weight = np.squeeze(np.array(params[-2].data.cpu()))#-2

#mean=np.mean(weight)
#new_weight=weight
#[r,c]=np.shape(weight)
#for j in range(0,r):
#    for k in range(0,c):
#        if weight[j,k] < 0.1:
#            new_weight[j,k]=0
#weight=new_weight

IMG_URL='/home/daa/Desktop/Grace_ML_Labs/cheng_dataset_test/test/'
masks_URL='/home/daa/Desktop/Grace_ML_Labs/cheng_dataset_test/test_masks/'

img_list=glob.glob(IMG_URL)
#fname='17'
org_loc=IMG_URL
predicted_labels=[]

for fname in range(1):#IMG_URL:
    with torch.no_grad():
        fname = '25.png'#{2,7,8,10,17,25,27,28,31,39,42} 
        img_pil = Image.open(org_loc+fname)
        #img_pil = cv2.imread(org_loc+fname)
        img_tensor = preprocess(img_pil)
        img_variable = Variable(img_tensor.unsqueeze(0)).cuda()
        logit = model(img_variable).cuda()

        h_x = F.softmax(logit, dim=1).data.squeeze()
 
        probs, idx = h_x.sort(0, True)
        probs = probs.detach().cpu().numpy()
        #probs = probs.cpu().numpy()
        idx = idx.cpu().numpy()
    
        predicted_labels.append(idx[0])
        predicted =  train_loader.dataset.classes[idx[0]]
    
        print("Target: " + fname + " | Predicted: " +  predicted) 
 
        features_blobs = mod(img_variable)
        features_blobs1 = features_blobs.detach().cpu().numpy()
        #print(np.shape(features_blobs1))
    
#---- comment either one ------------------------------------------------------    
        CAMs = return_pruned_CAM(params, features_blobs1, weight, [idx[0]], norm=2, PR=0)
        #CAMs = return_CAM(features_blobs1, weight, [idx[0]])
        
        EigenCAMs = EigenCAM(model, [model[-2][-1].myconv1])
        grayscale_cam = EigenCAMs(input_tensor=torch.unsqueeze(img_tensor,0), targets=None)
        grayscale_cam=np.squeeze(grayscale_cam,0)
        CAMs=[grayscale_cam,grayscale_cam,grayscale_cam]
        CAMs=np.uint8(255*np.asarray(CAMs))
#------------------------------------------------------------------------------
        readImg = org_loc+fname
        img = cv2.imread(readImg)
        height, width, _ = img.shape
        heatmap = cv2.applyColorMap(cv2.resize(CAMs[0],(width, height)), cv2.COLORMAP_JET)
        result = heatmap * 0.5 + img * 0.5
        cv2.imwrite('result.jpg',result)

#------------------------------------------------------------------------------    
        mask = cv2.imread(masks_URL+fname)
        _,binary = cv2.threshold(mask,0,255,cv2.THRESH_BINARY )
        edged = cv2.Canny(binary, 30, 200)
        #idx = cv2.findContours(binary,cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)#[1][0]
        contours, hierarchy = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE) 
    
        #out = result
        #out[idx[:,0,0],idx[:,0,1]] = 255
    
        cv2.drawContours(result,contours,-1,(0,0,0),2)
        cv2.putText(result, 'true=m, predicted=m', (100,500), cv2.FONT_HERSHEY_SIMPLEX,  
                   1, (0,255,255), 2, cv2.LINE_AA) 
        cv2.imwrite('caso1.jpg',result)            #Save the image
        #cv2.destroyAllWindows()
    













