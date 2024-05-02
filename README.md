# Explainable_brain_tumor_detection_in_MRI
Classification networks pre-trained on ImageNet and RadImageNet for expainable brain tumor detection on the Cheng et al. dataset

Cheng et al. dataset can be downloaded from:
https://figshare.com/articles/brain_tumor_dataset/1512427

Just run *train_XAIMed_Net.py* training script where you can select the corresponding model

Then run *compute_metrics.py*

Proposed Architecture of the **XAIMed-Net**:

![image](https://github.com/juliadietlmeier/Explainable_brain_tumor_detection_in_MRI/assets/79544193/caf5edb9-cb42-41b0-8db0-e035d7210a5f)


Qualitative results so far:

![image](https://github.com/juliadietlmeier/Explainable_brain_tumor_detection_in_MRI/assets/79544193/a9d4ca60-872d-4fbd-bd8a-79ebcd4c1c1e)


Second row: SEA1 ImageNet

Third row: SEA3 RadImageNet

Last row: B2-Net trained from scratch

Heatmap resolution (B0) = 14 x 14

Heatmap resolution (B1) = 14 x 14

heatmap resolution (B2) = 107 x 107
