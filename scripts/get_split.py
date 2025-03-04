import os
import shutil
import random

from tqdm import tqdm

# 设置随机种子以确保结果可复现
random.seed(42)

# 原始文件夹路径
images_folder = "/media/ubuntu/maxiaochuan/SAM_adaptive_learning/processed_data/Pancreas_MRI/images"
labels_folder = "/media/ubuntu/maxiaochuan/SAM_adaptive_learning/processed_data/Pancreas_MRI/labels"
out_image_folder = "/media/ubuntu/maxiaochuan/SAM_adaptive_learning/sam_data/Pancreas_MRI/images" 
out_label_folder = "/media/ubuntu/maxiaochuan/SAM_adaptive_learning/sam_data/Pancreas_MRI/labels" 

# 目标文件夹路径
output_folders = {
    "train": {"images": os.path.join(out_image_folder, "train"), "labels": os.path.join(out_label_folder, "train")},
    "valid": {"images": os.path.join(out_image_folder, "valid"), "labels": os.path.join(out_label_folder, "valid")},
    "test": {"images": os.path.join(out_image_folder, "test"), "labels": os.path.join(out_label_folder, "test")},
}

# 创建目标文件夹
for subset in output_folders.values():
    for folder in subset.values():
        os.makedirs(folder, exist_ok=True)

# 获取文件列表（假设 images 和 labels 文件夹下文件名一致）
image_files = sorted(os.listdir(images_folder))
label_files = sorted(os.listdir(labels_folder))

# 确保文件名一致
assert len(image_files) == len(label_files), "The number of images and labels must be the same."
assert all(os.path.splitext(image_files[i])[0] == os.path.splitext(label_files[i])[0] for i in range(len(image_files))), "File names in images and labels folders do not match."

# 打乱文件顺序
data = list(zip(image_files, label_files))
random.shuffle(data)

# 按比例划分数据集
total_files = len(data)
train_split = int(0.7 * total_files)
valid_split = int(0.1 * total_files)

train_data = data[:train_split]
valid_data = data[train_split:train_split + valid_split]
test_data = data[train_split + valid_split:]

# 定义辅助函数
def move_files(data_subset, subset_name):
    for image_file, label_file in tqdm(data_subset):
        # 源文件路径
        src_image_path = os.path.join(images_folder, image_file)
        src_label_path = os.path.join(labels_folder, label_file)
        
        # 目标文件路径
        dst_image_path = os.path.join(output_folders[subset_name]["images"], image_file)
        dst_label_path = os.path.join(output_folders[subset_name]["labels"], label_file)
        
        # 移动文件
        shutil.copy(src_image_path, dst_image_path)
        shutil.copy(src_label_path, dst_label_path)

# 移动文件到对应文件夹
move_files(train_data, "train")
move_files(valid_data, "valid")
move_files(test_data, "test")

print("Data split completed.")
