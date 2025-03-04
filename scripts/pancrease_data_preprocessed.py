import sys

from tqdm import tqdm
sys.path.append("/media/ubuntu/maxiaochuan/SAM_adaptive_learning")
import os
import SimpleITK as sitk
import numpy as np
from scipy.ndimage import zoom
from PIL import Image
import image_io

# 输入文件夹路径
base_folder = '/media/ubuntu/maxiaochuan/SAM_adaptive_learning/sam_data/UnresizedUTAH'
output_base_folder = '/media/ubuntu/maxiaochuan/SAM_adaptive_learning/sam_data/UTAH'
folders = ['train', 'valid']

# 缩放到 256x256 的目标尺寸
target_size = (512, 512)

# 处理每种数据类型
for data_type in ['images', 'labels']:
    # 处理每个子文件夹
    for folder in folders:
        input_folder = os.path.join(base_folder, data_type, folder)
        output_folder = os.path.join(output_base_folder, data_type, folder)

        # 创建输出文件夹
        os.makedirs(output_folder, exist_ok=True)

        # 遍历当前文件夹中的所有文件
        for file_name in tqdm(os.listdir(input_folder)):
            if not file_name.endswith('.nii.gz'):
                continue

            # 加载文件
            input_path = os.path.join(input_folder, file_name)
            input_array = image_io.read_nii(input_path)

            tqdm.write(f"Processing {file_name} in {data_type}/{folder}")
            
            # 计算缩放因子
            scale_factors = (
                target_size[0] / input_array.shape[0],
                target_size[1] / input_array.shape[1],
            )
            
            # 使用不同的插值方式
            if data_type == 'images':
                resized_array = zoom(input_array, scale_factors, order=0)  # 双线性插值
            else:
                resized_array = zoom(input_array, scale_factors, order=0)  # 最近邻插值
            
            # 保存处理后的文件
            output_path = os.path.join(output_folder, file_name)
            print(resized_array.shape)
            image_io.save_nii(resized_array, output_path)

print("Processing complete.")
