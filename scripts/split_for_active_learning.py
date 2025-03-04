import os
import shutil
import random

# 定义源目录和目标目录
source_dir = '/media/ubuntu/maxiaochuan/SAM_adaptive_learning/processed_data/Promise12/labels/train'
target_base_dir = '/media/ubuntu/maxiaochuan/SAM_adaptive_learning/processed_data/Promise12/labels'

# 创建目标目录
os.makedirs(target_base_dir, exist_ok=True)
for i in range(1, 6):
    os.makedirs(os.path.join(target_base_dir, f'folder_{i}'), exist_ok=True)

# 获取所有文件
files = [f for f in os.listdir(source_dir) if os.path.isfile(os.path.join(source_dir, f))]

# 打乱文件列表
random.seed(12)
random.shuffle(files)

# 计算每个文件夹的文件数量
total_files = len(files)
files_per_folder = total_files // 5

# 将文件分配到不同的文件夹
for i in range(5):
    start_index = i * files_per_folder
    if i == 4:  # 最后一个文件夹包含剩余的文件
        end_index = total_files
    else:
        end_index = start_index + files_per_folder
    for file in files[start_index:end_index]:
        shutil.copy2(os.path.join(source_dir, file), os.path.join(target_base_dir, f'folder_{i + 1}', file))

print("文件已分为五份并复制到不同的文件夹中。")
