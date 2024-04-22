"""
Created on Sat Apr 20 07:32:18 2024

@author: <julia.dietlmeier@insight-centre.org>
"""

import numpy as np

def dice(pred, true, k = 1):
    intersection = np.sum(pred[true==k]) * 2.0
    dice = intersection / (np.sum(pred) + np.sum(true))
    return dice

def seg_acc(pred_mask, gt_mask):
    eps = 1e-10
    p = pred_mask.astype(int) # predicted for test1
    g = gt_mask # gt for test1
        
    pvec=np.reshape(p, p.shape[0]*p.shape[1])
    gvec=np.reshape(g, g.shape[0]*g.shape[1])

    TP_idx = np.where((pvec==1) & (gvec==1)) 
    sh1=np.shape(TP_idx) 
    TP = sh1[1] 

    FP_idx = np.where((pvec==1) & (gvec==0)) 
    sh2=np.shape(FP_idx) 
    FP = sh2[1] 

    TN_idx = np.where((pvec==0) & (gvec==0)) 
    sh3=np.shape(TN_idx) 
    TN = sh3[1] 

    FN_idx = np.where((pvec==0) & (gvec==1)) 
    sh4=np.shape(FN_idx) 
    FN = sh4[1] 

    seg_acc = (TP+TN)/(TP+TN+FP+FN+eps)*100
    
    precision = TP/(TP + FP + eps)# avoid division by zero
    recall    = TP/(TP + FN + eps) # avoid division by zero
    Fscore = 2*( (precision * recall) / (precision + recall + eps) )# avoid division by zero
    
    Jaccard = TP/(TP+FP+FN + eps) 
    
    # Dice similarity function


    DSC = dice(pred_mask, gt_mask, k = 1) #can be k=255 
    
    return seg_acc, precision, recall, Fscore, Jaccard, DSC
