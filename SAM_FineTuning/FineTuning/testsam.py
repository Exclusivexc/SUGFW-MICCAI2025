from train import load_sam
from sklearn.preprocessing import binarize
import torch.nn as nn 
import logging
import torch.backends.cudnn as cudnn
import os
import numpy as np
import torch.optim as optim
import argparse
import random
import sys

from tqdm import tqdm
sys.path.append("..")
sys.path.append("/media/ubuntu/maxiaochuan/myscripts/data_augmentation")
sys.path.append("/media/ubuntu/maxiaochuan/ISICDM 2024/scripts")
from segment_anything_finetune import sam_model_registry
from find_all_bounding_box_2d import find_bounding_boxes
import image_io
from segment_anything_finetune.utils.transforms import ResizeLongestSide
from dataloader.dataloader import get_data_loader, RandomCrop, RandomRotFlip, ToTensor
from torchvision import transforms
import torch


class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-5):
        super(DiceLoss, self).__init__()
        self.smooth = smooth  # 防止除零错误，平滑参数

    def forward(self, preds, targets):
        # 将 logits 转为概率（0到1之间的值），应用 sigmoid

        # 计算交集和并集
        intersection = torch.sum(preds * targets)  # 交集
        union = torch.sum(preds * preds) + torch.sum(targets * targets)  # 并集
        # 计算 Dice 系数
        dice = (2. * intersection + self.smooth) / (union + self.smooth)  # 加上平滑项避免除零

        # Dice Loss = 1 - Dice Coefficient
        dice_loss = 1 - dice
        return dice_loss


def config():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--model_type', type=str, default='vit_b', help='sam checkpoint type')
    parser.add_argument('--root_path', type=str, default='/media/ubuntu/maxiaochuan/ISICDM 2024/SAM_FineTuning/FineTuning/pencoder_and_decoder', help='Name of Experiment')
    parser.add_argument('--save_model_name', type=str, default='pencoder_and_decoder')
    parser.add_argument('--max_epoch', type=int, default=1000, help='maximum epoch number to train')
    parser.add_argument('--batch_size', type=int, default=3, help='batch_size per gpu')
    parser.add_argument('--deterministic', type=int,  default=1, help='whether use deterministic training')
    parser.add_argument('--base_lr', type=float,  default=0.01, help='segmentation network learning rate')
    parser.add_argument('--patch_size', type=list,  default=[1024, 1024], help='patch size of network input')
    parser.add_argument('--seed', type=int,  default=2023, help='random seed')
    
    return parser.parse_args()


sam = load_sam(model_type="auto_lora_finetune")
with open("/media/ubuntu/maxiaochuan/ISICDM 2024/SAM_FineTuning/FineTuning/lora_parameters.txt", 'w') as f:
    for param_name, param in sam.named_parameters():
        f.write(f"{param_name}: {param.requires_grad} \n") 
    
# image_path = "/media/ubuntu/maxiaochuan/ISICDM 2024/nnUNet-master/nnUNet_raw/Dataset501_Polyp/imagesTr_RGB/Polyp_0365.png"
# # label_path = image_path.replace("images", "labels")
# pred_path = "/media/ubuntu/maxiaochuan/ISICDM 2024/nnUNet-master/nnUNet_results/Dataset501_Polyp/nnUNetTrainer__nnUNetPlans__2d_aug2_1024_1280/fold_0/poor_case_refine/Polyp_0365.png"


# image = image_io.read_png(image_path)
# pred = image_io.read_png(pred_path)
# batched_input = []


# transform = transforms.Compose([
#         ToTensor(),
#     ])


# with torch.no_grad():
#     data = {"image": image, "label": pred}
#     data = transform(data)
#     image = data["image"]
#     pred = data["label"]
#     original_size = tuple(image.shape[-2:])
#     sam_transform = ResizeLongestSide(sam.image_encoder.img_size)
#     image = image.permute(1, 2, 0).detach().cpu().numpy()
#     image = sam_transform.apply_image(image)
#     image = torch.as_tensor(image, dtype=torch.float).permute(2, 0, 1)
#     box = find_bounding_boxes(pred)[0]
#     box = np.array([box[1], box[0], box[3], box[2]])
#     box = sam_transform.apply_boxes(box, original_size=original_size)
#     box = torch.Tensor(box).cuda()
#     image = image.cuda()
#     pred = pred.cuda()
#     pred[pred > 0] = 1
#     batched_input.append({
#         'image': image,
#         'original_size': original_size,
#         'boxes': box,
#     })

#     outputs = sam(
#         batched_input=batched_input,
#         multimask_output=False,
#     )
#     outputs = torch.cat([x["masks"] for x in outputs], dim=0)
#     outputs = torch.sigmoid(outputs)
    
#     print(outputs.shape, pred.shape)
#     dice = DiceLoss()
#     d = 1 - dice(outputs, pred.unsqueeze(0)) 
#     print(d.item())
#     output = torch.zeros_like(outputs)
#     output[outputs > 0.5] = 1
#     output = output.squeeze().detach().cpu().numpy()
#     output = (output * 255).astype(np.uint8)
#     image_io.save_png(output, "/media/ubuntu/maxiaochuan/ISICDM 2024/SAM_FineTuning/FineTuning/test.png") 
