import os
import shutil
import numpy as np
import SimpleITK as sitk
from tqdm import tqdm

# 源文件夹路径
source_image_dir = '/media/ubuntu/maxiaochuan/SAM_adaptive_learning/raw_data/Promise12/slice/image'
source_label_dir = '/media/ubuntu/maxiaochuan/SAM_adaptive_learning/raw_data/Promise12/slice/label'

# 目标文件夹路径
target_dir = '/media/ubuntu/maxiaochuan/SAM_adaptive_learning/sam_data/Promise12'

# 创建目标文件夹结构
os.makedirs(os.path.join(target_dir, 'image/train'), exist_ok=True)
os.makedirs(os.path.join(target_dir, 'image/valid'), exist_ok=True)
os.makedirs(os.path.join(target_dir, 'image/test'), exist_ok=True)
os.makedirs(os.path.join(target_dir, 'label/train'), exist_ok=True)
os.makedirs(os.path.join(target_dir, 'label/valid'), exist_ok=True)
os.makedirs(os.path.join(target_dir, 'label/test'), exist_ok=True)

# 函数：检查文件是否为空
def is_not_empty(file_path):
    t = sitk.GetArrayFromImage(sitk.ReadImage(file_path))
    return t.sum()


# 移动文件
for split in ['train', 'valid', 'test']:
    source_image_split_dir = os.path.join(source_image_dir, split)
    source_label_split_dir = os.path.join(source_label_dir, split)
    target_image_split_dir = os.path.join(target_dir, 'image', split)
    target_label_split_dir = os.path.join(target_dir, 'label', split)

    for filename in tqdm(os.listdir(source_image_split_dir)):
        # 构建完整的文件路径
        image_path = os.path.join(source_image_split_dir, filename)
        label_path = os.path.join(source_label_split_dir, filename)

        # 检查标签文件是否存在且不为空
        if os.path.exists(label_path) and is_not_empty(label_path):
            # 移动图像文件
            shutil.move(image_path, target_image_split_dir)
            # 移动标签文件
            shutil.move(label_path, target_label_split_dir)

print("文件移动完成。")
