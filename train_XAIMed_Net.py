#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 20 07:32:18 2024

@author: <julia.dietlmeier@insight-centre.org>
"""
import cv2
import torch
import torch.nn as nn
import numpy as np
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
from tqdm.auto import tqdm
from models import resnet50only, SEA3_resnet50, SEA2_resnet50, SEA1_resnet50, B2_Net, SEA3_resnet50_RadImageNet
import matplotlib.pyplot as plt

device="cuda"

seed = 42
if device == "cuda":
    torch.cuda.manual_seed_all(seed)  # GPU seed
    torch.backends.cudnn.deterministic = True  # fix the GPU to deterministic mode
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


class EarlyStopper:
    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = np.inf

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False


"=============================================================================="
batch_size=16
num_epochs=100

normalize = transforms.Normalize(
   mean=[0.485, 0.456, 0.406],
   std=[0.229, 0.224, 0.225]
)

img_transform = transforms.Compose([transforms.Resize((224, 224)),transforms.ToTensor(),
                                    normalize])

train_data_dir = datasets.ImageFolder("/home/daa/Desktop/Grace_ML_Labs/cheng_dataset_Julias_split/train/", transform = img_transform)

train_loader = torch.utils.data.DataLoader(train_data_dir, batch_size=batch_size, shuffle=True)

val_data_dir = datasets.ImageFolder("/home/daa/Desktop/Grace_ML_Labs/cheng_dataset_Julias_split/val/", transform = img_transform)

val_loader = torch.utils.data.DataLoader(val_data_dir, batch_size=batch_size, shuffle=False)

"=== MODEL ===================================================================="
#model = SEA3_resnet50().cuda()
model = SEA1_resnet50().cuda()
#model = SEA3_resnet50_RadImageNet().cuda()
#model=B2_Net().cuda()

trainable_parameters = []
for name, p in model.named_parameters():
    if "fc" or "myconv1" or "myconv2" or "myconv3" or "myconv4" or "myconv5" or "myconv6" or "att1" or "att2" or "att3" or "att4" or "att5" or "att6" in name:
        trainable_parameters.append(p)

y1=np.zeros((998))#number of glioma images
y2=np.ones((496))#number of meningioma images
y3=2*np.ones((651))#number of pituitary images
y_train=np.concatenate((y1,y2,y3))
from sklearn.utils import class_weight
class_weights = class_weight.compute_class_weight(class_weight='balanced',
                                                 classes=np.unique(y_train),
                                                 y=y_train)

class_weights = torch.FloatTensor(class_weights).cuda()

optimizer = torch.optim.AdamW(params=trainable_parameters, lr=0.00001) 
 
criterion = nn.CrossEntropyLoss(weight=class_weights)

"=== TRAIN ===================================================================="

total_step = len(train_loader)
loss_list = []
acc_list = []
loss_arr=[]
val_loss_arr=[]
val_acc_list=[]
val_loss_list=[]
training_loss=[]
validation_loss=[]
validation_acc=[]
training_acc=[]

early_stopper = EarlyStopper(patience=2, min_delta=0.5)

for epoch in range(num_epochs):

    for i, data in tqdm(enumerate(train_loader),total=len(train_loader)):
        images,labels = data
        images=images.cuda()
        labels=labels.cuda()
        # Run the forward pass
        outputs = model(images)
        
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
        acc=correct / total
        acc_list.append(acc)

        if (i + 1) % 100 == 0:
            print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Accuracy: {:.2f}%'
                  .format(epoch + 1, num_epochs, i + 1, total_step, loss.item(),
                          (correct / total) * 100))
            loss_arr.append(loss.item())
            training_acc.append(((correct / total) * 100))
         
    for i, data in tqdm(enumerate(val_loader), total=len(val_loader)):
        model.eval()
        with torch.no_grad():
            images,labels = data
            images=images.cuda()
            labels=labels.cuda()
         # Run the forward pass
            outputs = model(images)
         #print(outputs.shape)
            val_loss = criterion(outputs, labels)
            val_loss_list.append(val_loss.item())

         # Track the accuracy
            total = labels.size(0)
            _, predicted = torch.max(outputs.data, 1)
            correct = (predicted == labels).sum().item()
            val_acc_list.append(correct / total)

            if (i + 1) % 10 == 0:
                print('Epoch [{}/{}], Step [{}/{}], Validation Loss: {:.4f}, Validation Accuracy: {:.2f}%'
                      .format(epoch + 1, num_epochs, i + 1, total_step, val_loss.item(),
                          (correct / total) * 100))
                val_loss_arr.append(val_loss.item())       
                validation_acc.append(((correct / total) * 100))
    
    if early_stopper.early_stop(loss): # prevent training loss from exploding            
        break           
    
    save_best_model(val_loss, epoch, model, optimizer, criterion)
    validation_loss.append(np.mean(val_loss_arr))
    training_loss.append(np.mean(loss_arr))
        
torch.save(model.state_dict(), 'SEA3_v2_resnet50_100epochs_ImageNet.pt')

plt.figure(),plt.plot(training_loss,'-k',label='training_loss'),plt.plot(validation_loss,'--r',label='validation_loss'),plt.legend(),plt.title('Loss history')
plt.figure(),plt.plot(training_acc,'-k',label='training_accuracy'),plt.plot(validation_acc,'--r',label='validation_accuracy'),plt.legend(),plt.title('Accuracy history')





