import os
import numpy as np
from PIL import Image

# 设置文件夹路径
image_folder = '/media/ubuntu/maxiaochuan/SAM_adaptive_learning/raw_data/Promise12/images'
save_folder = "/media/ubuntu/maxiaochuan/SAM_adaptive_learning/raw_data/Promise12/images_normalized"
# 遍历文件夹中的所有 PNG 图像
for filename in os.listdir(image_folder):
    if filename.endswith('.png'):
        # 加载图像
        image_path = os.path.join(image_folder, filename)
        img = Image.open(image_path)

        # 将图像转换为 numpy 数组
        img_array = np.array(img)

        # 计算均值和标准差
        mean = np.mean(img_array)
        std = np.std(img_array)

        # 对图像进行归一化
        normalized_img_array = (img_array - mean) / std

        # 将归一化后的数组转换回图像
        normalized_img = Image.fromarray(np.uint8(np.clip(normalized_img_array * 255, 0, 255)))

        # 保存归一化后的图像
        normalized_image_path = os.path.join(save_folder, filename)
        normalized_img.save(normalized_image_path)
