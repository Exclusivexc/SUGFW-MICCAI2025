import os
import shutil
import random

from tqdm import tqdm


def split_five_Fold():
    # 定义源目录和目标目录
    source_images_dir = '/media/ubuntu/maxiaochuan/SAM_adaptive_learning/raw_data/Promise12/images'
    source_labels_dir = '/media/ubuntu/maxiaochuan/SAM_adaptive_learning/raw_data/Promise12/labels'
    target_images_dir = '/media/ubuntu/maxiaochuan/SAM_adaptive_learning/processed_data/Promise12/images'
    target_labels_dir = '/media/ubuntu/maxiaochuan/SAM_adaptive_learning/processed_data/Promise12/labels'

    # 创建目标目录
    os.makedirs(os.path.join(target_images_dir, 'train'), exist_ok=True)
    os.makedirs(os.path.join(target_images_dir, 'validation'), exist_ok=True)
    os.makedirs(os.path.join(target_images_dir, 'test'), exist_ok=True)

    os.makedirs(os.path.join(target_labels_dir, 'train'), exist_ok=True)
    os.makedirs(os.path.join(target_labels_dir, 'validation'), exist_ok=True)
    os.makedirs(os.path.join(target_labels_dir, 'test'), exist_ok=True)

    # 获取所有样本文件
    image_files = [f for f in os.listdir(source_images_dir) if os.path.isfile(os.path.join(source_images_dir, f))]
    label_files = [f for f in os.listdir(source_labels_dir) if os.path.isfile(os.path.join(source_labels_dir, f))]
    print(len(image_files), len(label_files))
    # 确保图像和标签文件一一对应
    assert len(image_files) == len(label_files), "图像和标签文件数量不匹配"

    # 打乱文件列表
    combined_files = list(zip(image_files, label_files))
    # print(combined_files)
    random.shuffle(combined_files)

    # 划分比例
    total_files = len(combined_files)
    train_size = int(total_files * 0.7)
    validation_size = int(total_files * 0.1)

    # 划分文件
    train_files = combined_files[:train_size]
    validation_files = combined_files[train_size:train_size + validation_size]
    test_files = combined_files[train_size + validation_size:]

    # 复制训练文件
    # exit()
    for image_file, label_file in train_files:
        shutil.copy2(os.path.join(source_images_dir, image_file), os.path.join(target_images_dir, 'train', image_file))
        shutil.copy2(os.path.join(source_labels_dir, label_file), os.path.join(target_labels_dir, 'train', label_file))

    # 复制验证文件
    for image_file, label_file in validation_files:
        shutil.copy2(os.path.join(source_images_dir, image_file), os.path.join(target_images_dir, 'validation', image_file))
        shutil.copy2(os.path.join(source_labels_dir, label_file), os.path.join(target_labels_dir, 'validation', label_file))

    # 复制测试文件
    for image_file, label_file in test_files:
        shutil.copy2(os.path.join(source_images_dir, image_file), os.path.join(target_images_dir, 'test', image_file))
        shutil.copy2(os.path.join(source_labels_dir, label_file), os.path.join(target_labels_dir, 'test', label_file))

    print("样本划分和复制完成。")




def split_files_to_folders(src_dir, num_folders, tag_dir, percentage=0.01, seed=None):
    """
    将文件从源文件夹分割到多个子文件夹，每个子文件夹包含5%的随机文件。
    
    :param src_dir: 源文件夹路径
    :param num_folders: 生成的子文件夹数量
    :param percentage: 每个子文件夹包含的文件比例（默认为5%）
    :param seed: 随机种子
    """
    # 获取所有文件
    files = [f for f in os.listdir(src_dir) if os.path.isfile(os.path.join(src_dir, f))]
    
    # 确定每次选取文件的数量
    num_files_to_select = int(len(files) * percentage)
    
    # 为每个子文件夹生成文件
    for i in range(1, num_folders + 1):
        folder_name = f"folder{i}"
        folder_path = os.path.join(tag_dir, folder_name)
        
        # 如果文件夹不存在，则创建它
        os.makedirs(folder_path, exist_ok=True)
        os.makedirs(folder_path.replace("images", "labels"), exist_ok=True)
        # 设置不同的随机种子
        if seed is not None:
            random.seed(seed + i)  # 给每个子文件夹一个不同的种子

        # 随机选择文件
        selected_files = random.sample(files, num_files_to_select)

        # 将选中的文件复制到对应的子文件夹
        for file in tqdm(selected_files):
            src_file = os.path.join(src_dir, file)
            dest_file = os.path.join(folder_path, file)
            shutil.copy(src_file, dest_file)
            shutil.copy(src_file.replace("images", "labels"), dest_file.replace("images", "labels"))

# 示例使用
src_dir = "/media/ubuntu/maxiaochuan/SAM_adaptive_learning/processed_data/Promise12/images/train"  # 请替换为你的文件夹路径
tag_dir = "/media/ubuntu/maxiaochuan/SAM_adaptive_learning/processed_data/Promise12/images"
num_folders = 5  # 要生成的文件夹数量
split_files_to_folders(src_dir, num_folders, tag_dir)
