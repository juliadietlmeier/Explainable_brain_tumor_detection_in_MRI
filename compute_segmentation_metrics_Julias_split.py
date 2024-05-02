#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 24 08:43:07 2024

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
from score_cam import ScoreCAM
from skimage.morphology import erosion, dilation, opening, closing, white_tophat  # noqa
from skimage.morphology import black_tophat, skeletonize, convex_hull_image  # noqa
from skimage.morphology import disk  # noqa
from skimage.measure import regionprops
from models import resnet50only, SEA3_resnet50, SEA2_resnet50, SEA1_resnet50, B2_Net, SEA3_resnet50_RadImageNet
import glob
from seg_acc_metrics import seg_acc
import math
from apply_crf import apply_crf

device="cuda"
seed = 42
if device == "cuda":
    torch.cuda.manual_seed_all(seed)  # GPU seed
    torch.backends.cudnn.deterministic = True  # fix the GPU to deterministic mode
    torch.manual_seed(seed)  # CPU seed
    random.seed(seed)  # python seed for image transformation
    np.random.seed(seed)
    
# --- comment which model you are using ---------------------------------------
#model=resnet50only().cuda()
model=SEA3_resnet50().cuda()
#model = SEA3_resnet50_RadImageNet().cuda()
#model=B2_Net().cuda()
#model=SEA2_resnet50().cuda()
#model=SEA1_resnet50().cuda()

#state_dict = torch.load('/home/daa/Desktop/Grace_ML_Labs/Code/resnet50_100epochs_ImageNet.pt', map_location=torch.device(device))
state_dict = torch.load('/home/daa/Desktop/Grace_ML_Labs/Code/SEA3_v2_resnet50_100epochs_ImageNet.pt', map_location=torch.device(device))
#state_dict = torch.load('/home/daa/Desktop/Grace_ML_Labs/Code/best_model.pt', map_location=torch.device(device))
#state_dict = torch.load('/home/daa/Desktop/Grace_ML_Labs/Code/SEA2_resnet50_100epochs_ImageNet.pt', map_location=torch.device(device))
#state_dict = torch.load('/home/daa/Desktop/Grace_ML_Labs/Code/SEA1_resnet50_100epochs_ImageNet.pt', map_location=torch.device(device))

model.load_state_dict(state_dict, strict=False)

normalize = transforms.Normalize(
   mean=[0.485, 0.456, 0.406],
   std=[0.229, 0.224, 0.225]
)
preprocess = transforms.Compose([
   transforms.Resize((224,224)),
   transforms.ToTensor(),
   normalize
])


IMG_URL='/home/daa/Desktop/Grace_ML_Labs/cheng_dataset_Julias_split/test/'
masks_URL='/home/daa/Desktop/Grace_ML_Labs/cheng_dataset_Julias_split/test_masks_all/'
dst_dir='/home/daa/Desktop/Grace_ML_Labs/Code/SEA3_results/'

file_list=glob.glob(IMG_URL+ '/*.png')


img_list=glob.glob(IMG_URL)
org_loc=IMG_URL
predicted_labels=[]

seg_acc_arr=[] 
precision_arr=[] 
recall_arr=[] 
Fscore_arr=[] 
Jaccard_arr = [] 
DSC_arr = []
test_acc = []
distance_arr=[]
distance_arr_centroid=[]
distance_arr_glioma=[]
distance_arr_meningioma=[]
distance_arr_pituitary=[]

normalize = transforms.Normalize(
   mean=[0.485, 0.456, 0.406],
   std=[0.229, 0.224, 0.225]
)


batch_size=1

img_transform = transforms.Compose([transforms.Resize((224, 224)),transforms.ToTensor(),
                                    normalize])
test_data_dir = datasets.ImageFolder("/home/daa/Desktop/Grace_ML_Labs/cheng_dataset_Julias_split/test/", transform = img_transform)

test_loader = torch.utils.data.DataLoader(test_data_dir, batch_size=batch_size, shuffle=False)

for i, data in tqdm(enumerate(test_loader), total=len(test_loader)):
    model.eval()
    with torch.no_grad():      
        
        images,labels = data
        images=images.cuda()
        labels=labels.cuda()
        logit = model(images)
        fname, _ = test_loader.dataset.samples[i]
        print(fname)
        h_x = F.softmax(logit, dim=1).data.squeeze()
  
        probs, idx = h_x.sort(0, True)
        probs = probs.detach().cpu().numpy()

        idx = idx.cpu().numpy()
    
        predicted_labels.append(idx[0])
        predicted =  test_loader.dataset.classes[idx[0]]
        
        GT_text = fname.split('/')[-1]
        GT_text = GT_text.split('.')[0]
        GT_text = GT_text.split('_')[-1]
        
        if predicted == 'glioma':
            predicted_text = 'g'
        elif predicted == 'meningioma':
            predicted_text = 'm'
        elif predicted == 'pituitary':
            predicted_text = 'p'
        
        total = len(file_list)
        
        if predicted_text == GT_text:
            correct = 1
        else:
            correct = 0   
        
        print("Target: " + GT_text + " | Predicted: " +  predicted_text)  
#-------comment for each model not used ---------------------------------------        
        #EigenCAMs = EigenCAM(model, [model[-2][-1][-1].conv3])# resnet50only
        #EigenCAMs = EigenCAM(model, [model[-2][-1].myconv4])#SEA2
        #EigenCAMs = EigenCAM(model, [model.myconv3])#B2_Net        
        
        EigenCAMs = EigenCAM(model, [model[-2][-1].myconv6])#SEA3
        #EigenCAMs = EigenCAM(model, [model[-2][-2][-1][-1].conv3])#SEA3 last conv of resnet backbone
        
        #EigenCAMs = EigenCAM(model, [model[-2][-1].myconv2])#SEA1
#------------------------------------------------------------------------------        
        grayscale_cam = EigenCAMs(input_tensor=images, targets=None)
        grayscale_cam=np.squeeze(grayscale_cam,0)
        CAMs=[grayscale_cam,grayscale_cam,grayscale_cam]
        CAMs=np.uint8(255*np.asarray(CAMs))
#------------------------------------------------------------------------------
        readImg = fname
        img = cv2.imread(readImg)
        height, width, _ = img.shape
        heatmap = cv2.applyColorMap(cv2.resize(CAMs[0],(width, height)), cv2.COLORMAP_JET)        
        result = heatmap * 0.5 + img * 0.5
#------------------------------------------------------------------------------    
        fname_idx = fname.split('/')[-1]
        mask = cv2.imread(masks_URL+fname_idx)
        vals=np.unique(mask)
        low=np.min(vals)
        high=np.max(vals)
        _,binary = cv2.threshold(mask,low,high,cv2.THRESH_BINARY )
        edged = cv2.Canny(binary, 0, 255)
        contours, hierarchy = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)     
        M = cv2.moments(contours[0])
        cx1 = int(M['m10']/M['m00'])
        cy1 = int(M['m01']/M['m00'])
        
        cv2.drawContours(result,contours,-1,(0,0,0),2)
        
        cv2.putText(result, 'predicted='+predicted_text, (200,500), cv2.FONT_HERSHEY_SIMPLEX,  
                   1, [0,255,255], 2, cv2.LINE_AA) 
        cv2.putText(result, 'true='+GT_text, (50,500), cv2.FONT_HERSHEY_SIMPLEX,  
                   1, [0,255,255], 2, cv2.LINE_AA) 
        
        new_heatmap=np.tanh(np.array(heatmap[:,:,1]))-np.tanh(np.array(heatmap[:,:,2]))
        pred_mask = np.zeros((512,512))
        [r,col] = np.where(new_heatmap == -1)
        pred_mask[r,col]=1
        
        mask = cv2.resize(mask,(512, 512))
        mask = (1-mask[:,:,0]/255)
        
#-------CRF -------------------------------------------------------------------        
        footprint = disk(30)# was 6
        dilated = dilation(pred_mask, footprint)
        
        kX=45# was 45
        kY=45# was 45
        pred=cv2.GaussianBlur(dilated, (kX, kY), 0)# was 0
        probs = (pred-np.min(pred))/(np.max(pred)-np.min(pred))
        probs = np.tile(probs[np.newaxis,:,:],(2,1,1))
        probs[1,:,:] = 1 - probs[0,:,:]

        crf_result=1-apply_crf(cv2.resize(img,(512,512)), probs)
        #crf_result = pred_mask
#------- comment if not using CRF ---------------------------------------------
        
        #seg_accuracy, precision, recall, Fscore, Jaccard, DSC = seg_acc(pred_mask.astype(int), mask.astype(int))

        seg_accuracy, precision, recall, Fscore, Jaccard, DSC = seg_acc(crf_result.astype(int), mask.astype(int))

#------------------------------------------------------------------------------        
        seg_acc_arr.append(seg_accuracy)
        precision_arr.append(precision)
        recall_arr.append(recall)
        Fscore_arr.append(Fscore)
        Jaccard_arr.append(Jaccard)
        DSC_arr.append(DSC)
        
#------ compute delta(pred_mask, mask)-----------------------------------------
# ---- need to compute center of mass------------------------------------------
        
        img=cv2.resize(img,(512,512))
        properties_pred = regionprops(pred_mask.astype('uint8'), img)
        p1 = properties_pred[0].centroid
        properties_true = regionprops(mask.astype('uint8'), img)
        p2 = properties_true[0].centroid
        distance_centroid = math.sqrt(((p1[0] - p2[0]) ** 2) + ((p1[1] - p2[1]) ** 2))
        
#---- different way to compute the center -------------------------------------        
        eps=1e-8
        vals=np.unique(pred_mask)
        low=np.min(vals)
        high=np.max(vals)
        _,binary = cv2.threshold(pred_mask.astype('uint8'),low,high,cv2.THRESH_BINARY )
        edged = cv2.Canny(binary*255, 0, 255)
        contours2, hierarchy2 = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)     
        

        M = cv2.moments(contours2[0])
        cx2 = int(M['m10']/(M['m00']+eps))
        cy2 = int(M['m01']/(M['m00']+eps))
           
        distance = math.sqrt(((cx1 - cx2) ** 2) + ((cy1 - cy2) ** 2))
        distance_arr_centroid.append(distance_centroid)
        
        if predicted=='glioma':
            distance_arr_glioma.append(distance)
        elif predicted=='meningioma':
            distance_arr_meningioma.append(distance)
        elif predicted=='pituitary':
            distance_arr_pituitary.append(distance)
        
        distance_arr.append(distance)
        
        
        cv2.drawContours(result,contours,-1,(0,0,0),2)        
        cv2.putText(result, 'predicted='+predicted_text, (200,500), cv2.FONT_HERSHEY_SIMPLEX,  
                   1, [0,255,255], 2, cv2.LINE_AA) 
        cv2.putText(result, 'true='+GT_text, (50,500), cv2.FONT_HERSHEY_SIMPLEX,  
                   1, [0,255,255], 2, cv2.LINE_AA) 
        cv2.putText(result, 'delta='+str(round(distance,2)), (50,450), cv2.FONT_HERSHEY_SIMPLEX,  1, [0,255,255], 2, cv2.LINE_AA)
        cv2.imwrite(dst_dir+fname_idx,result)            #Save the image
        
    test_acc.append(correct)

classification_test_acc = np.mean(np.asarray(test_acc))*100
seg_accuracy=np.mean(np.asarray(seg_acc_arr))
precision=np.mean(np.asarray(precision_arr))
recall=np.mean(np.asarray(recall_arr))
Fscore=np.mean(np.asarray(Fscore_arr))
Jaccard=np.mean(np.asarray(Jaccard_arr))
dice=np.mean(np.asarray(DSC_arr))

delta_moment=np.mean(np.asarray(distance_arr))
delta_moment_glioma=np.mean(np.asarray(distance_arr_glioma))
delta_moment_meningioma=np.mean(np.asarray(distance_arr_meningioma))
delta_moment_pituitary=np.mean(np.asarray(distance_arr_pituitary))
delta_centroid=np.mean(np.asarray(distance_arr_centroid))



















