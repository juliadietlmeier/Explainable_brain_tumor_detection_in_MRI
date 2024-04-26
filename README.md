# Explainable_brain_tumor_detection_in_MRI
Classification networks pre-trained on ImageNet and RadImageNet for expainable brain tumor detection on the Cheng et al. dataset

Cheng et al. dataset can be downloaded from:
https://figshare.com/articles/brain_tumor_dataset/1512427

*ImageNet pretrained Backbone 0: CAMs_resnet50.py*

*RadImageNet pretrained Backbone 0: CAMs_resnet50_RadImageNet.py*

*B2 trained from scratch: CAMs_B2_from_scratch.py*

Proposed Architecture of the **XAIMed-Net**:

![image](https://github.com/juliadietlmeier/Explainable_brain_tumor_detection_in_MRI/assets/79544193/76bfe117-929d-43f5-95aa-7f674cb920d5)


Qualitative results so far:

![image](https://github.com/juliadietlmeier/Explainable_brain_tumor_detection/assets/79544193/5c7417a6-0d2e-41d5-86ef-368635ca2bc6)


We finetune B0 and B1. We train from scratch B2

Heatmap resolution (B0) = 14 x 14

Heatmap resolution (B1) = 14 x 14

heatmap resolution (B2) = 107 x 107
