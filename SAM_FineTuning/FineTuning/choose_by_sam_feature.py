import json
import os
import sys

from tqdm import tqdm
sys.path.append("/media/ubuntu/maxiaochuan/SAM_adaptive_learning")
sys.path.append("/media/ubuntu/maxiaochuan")
import torch
from segment_anything_finetune import sam_model_registry, SamAutomaticMaskGenerator
from augmentation.intensity_and_contrast import *
from augmentation.normalization import Range_Norm
from PIL import Image
import matplotlib.pyplot as plt
import config
import numpy as np
import SimpleITK as sitk
from numpy import ndarray
from typing import *
from colorsys import hls_to_rgb



def get_iou(pred: ndarray, mask: ndarray) -> float:
    pred_positives = pred.sum()
    mask_positives = mask.sum()
    inter = (pred * mask).sum()
    union = pred_positives + mask_positives
    iou = inter / (union - inter + 1e-6)
    return iou



""" load model """
print("loading model.....")
model_type = "vit_b"  # 根据需求选择模型类型，如 'vit_h', 'vit_b' 等
checkpoint = config.sam_vit_b_path  # 替换为下载的模型权重路径

sam = sam_model_registry[model_type](checkpoint=checkpoint)
sam = sam.to(device="cuda:3" if torch.cuda.is_available() else "cpu")

mask_generator = SamAutomaticMaskGenerator(sam)
print(f"successfully load sam {model_type}!")

""" load data """
image_dir = "/media/ubuntu/maxiaochuan/SAM_adaptive_learning/raw_data/Promise12/slice/image/train"
alls = []

files = os.listdir(image_dir)
for file in tqdm(files):
    # print(file)
    image_path = os.path.join(image_dir, file)
    label_path = image_path.replace("image", "label")
    image = sitk.GetArrayFromImage(sitk.ReadImage(image_path)).squeeze() # (1, 320, 320) -> (320, 320)
    image = Range_Norm(image, [0, 255])
    image = image.astype(np.uint8)
    aug_image = image.copy()
    aug_image = contrast_and_bightness_enhancing(image=aug_image)
    image = np.expand_dims(image, axis=2).repeat(3, axis=2)
    aug_image = np.expand_dims(aug_image, axis=2).repeat(3, axis=2)
    mask = get_mask(image)
    mask_aug = get_mask(aug_image)
    iou = get_iou(mask, mask_aug)
    save_mask(mask, f"/media/ubuntu/maxiaochuan/SAM_adaptive_learning/SAM_choose_sample/results/mask/{file}.png")
    save_mask(mask_aug, f"/media/ubuntu/maxiaochuan/SAM_adaptive_learning/SAM_choose_sample/results/mask_aug/{file}.png")
    alls.append([iou, image_path, label_path])

alls.sort()
alls.reverse()
file_name = '/media/ubuntu/maxiaochuan/SAM_adaptive_learning/SAM_choose_sample/results/iou.txt'
print(f"Writting data.............")
with open(file_name, 'w') as file:
    for iou, image_path, label_path in tqdm(alls):
        file.write(f"{iou}, {image_path}, {label_path}\n")

print(f"Data has been written to {file_name}")