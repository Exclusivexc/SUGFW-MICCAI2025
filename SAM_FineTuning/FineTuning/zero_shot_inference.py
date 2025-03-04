from medpy import metric
import os
import SimpleITK as sitk
import cv2
import numpy as np
import sys
sys.path.append("/media/ubuntu/maxiaochuan/myscripts")
sys.path.append("/media/ubuntu/maxiaochuan/ISICDM 2024/SAM_FineTuning")
from segment_anything_finetune import sam_model_registry, SamPredictor
from segment_anything_finetune.utils.transforms import ResizeLongestSide
from tqdm import tqdm
import image_io
from train import load_sam
import torch
import torch.nn as nn 


class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-5):
        super(DiceLoss, self).__init__()
        self.smooth = smooth  # 防止除零错误，平滑参数

    def forward(self, preds, targets):
        # 将 logits 转为概率（0到1之间的值），应用 sigmoid

        # 计算交集和并集
        intersection = np.sum(preds * targets)  # 交集
        union = np.sum(preds * preds) + np.sum(targets * targets)  # 并集
        # 计算 Dice 系数
        dice = (2. * intersection + self.smooth) / (union + self.smooth)  # 加上平滑项避免除零

        # Dice Loss = 1 - Dice Coefficient
        dice_loss = 1 - dice
        return dice_loss

sam = load_sam(model_type="vit_h")
predictor = SamPredictor(sam)

label_path = "/media/ubuntu/maxiaochuan/ISICDM 2024/SAM_FineTuning/data/SAM_labelsTr/6.png"
image_path = label_path.replace("labels", "images")
box_path = image_path.replace(".png", ".txt")
image = image_io.read_png(image_path)
label = image_io.read_png(label_path)
with open(box_path, 'r') as f:
    box = f.readline()

box = box.split(',')
box = np.array([box[1], box[0], box[3], box[2]])


predictor.set_image(image)
# input_point = np.array(foreground_points)
# input_label = np.array(label_input)

masks, _, _ = predictor.predict(
    box=box,
    # point_coords=input_point,

    # point_labels=input_label,
    multimask_output=False,
    return_logits=True,
)
loss = DiceLoss()
print(masks)
masks = torch.sigmoid(torch.from_numpy(masks)).detach().cpu().numpy()
print(loss(masks, label))

# image_io.save_png(masks, "/media/ubuntu/maxiaochuan/test.png")
   
