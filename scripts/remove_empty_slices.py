import os
import numpy as np
from glob import glob
from tqdm import tqdm
import shutil
import SimpleITK as sitk

def process_single_split(base_dir, output_dir, split):
    """处理单个数据集分割（train/valid/test）"""
    # 设置路径
    label_dir = os.path.join(base_dir, "labels", split)
    image_dir = os.path.join(base_dir, "images", split)
    
    # 设置输出路径
    out_image_dir = os.path.join(output_dir, "images", split)
    out_label_dir = os.path.join(output_dir, "labels", split)
    
    os.makedirs(out_image_dir, exist_ok=True)
    os.makedirs(out_label_dir, exist_ok=True)
    
    # 获取所有label文件
    label_files = sorted(glob(os.path.join(label_dir, "*.nii.gz")))
    
    if not label_files:
        print(f"Warning: No .nii.gz files found in {label_dir}")
        return 0, 0

    total = len(label_files)
    kept = 0
    
    for label_file in tqdm(label_files, desc=f"Processing {split}"):
        # 获取对应的图像文件名
        filename = os.path.basename(label_file)
        image_file = os.path.join(image_dir, filename)
        
        # 检查图像文件是否存在
        if not os.path.exists(image_file):
            print(f"Warning: Missing image file for {filename}")
            continue
            
        # 使用SimpleITK读取nii.gz文件
        try:
            label_img = sitk.ReadImage(label_file)
            label = sitk.GetArrayFromImage(label_img)
            
            # 如果标签不全为0，复制文件
            if np.sum(label) > 0:
                shutil.copy2(image_file, os.path.join(out_image_dir, filename))
                shutil.copy2(label_file, os.path.join(out_label_dir, filename))
                kept += 1
        except Exception as e:
            print(f"Error processing {filename}: {str(e)}")
            continue
    
    return total, kept

def copy_uncertainty_folder(base_dir, output_dir):
    """完整复制uncertainty文件夹"""
    source_dir = os.path.join(base_dir, "uncertainty")
    target_dir = os.path.join(output_dir, "uncertainty")
    
    if os.path.exists(source_dir):
        print("\nCopying uncertainty folder...")
        shutil.copytree(source_dir, target_dir, dirs_exist_ok=True)
        print("Uncertainty folder copied successfully")

def main():
    # 基础目录设置
    base_dir = "/media/ubuntu/maxiaochuan/SAM_adaptive_learning/sam_data/UTAH"
    output_dir = "/media/ubuntu/maxiaochuan/SAM_adaptive_learning/sam_data/UTAH_filtered"
    
    # 创建必要的目录
    for folder in ['images', 'labels']:
        for split in ["train", "valid", "test"]:
            os.makedirs(os.path.join(output_dir, folder, split), exist_ok=True)
    
    # 处理每个分割
    total_stats = {"total": 0, "kept": 0}
    
    for split in ["train", "valid", "test"]:
        print(f"\nProcessing {split} set...")
        total, kept = process_single_split(base_dir, output_dir, split)
        removed = total - kept
        
        print(f"Results for {split}:")
        print(f"- Total slices: {total}")
        print(f"- Kept slices: {kept}")
        print(f"- Removed slices: {removed}")
        
        total_stats["total"] += total
        total_stats["kept"] += kept
    
    # 复制完整的uncertainty文件夹
    copy_uncertainty_folder(base_dir, output_dir)
    
    # 检查总处理文件数
    if total_stats["total"] == 0:
        print("\nNo files were processed!")
        print("Please check if the input directories exist and contain .npy files.")
        return
        
    # 打印总体统计信息
    print("\nOverall statistics:")
    print(f"Total processed: {total_stats['total']}")
    print(f"Total kept: {total_stats['kept']}")
    print(f"Total removed: {total_stats['total'] - total_stats['kept']}")
    
    # 只在有处理文件时计算比率
    if total_stats["total"] > 0:
        kept_ratio = total_stats['kept']/total_stats['total']*100
        print(f"Kept ratio: {kept_ratio:.2f}%")

if __name__ == "__main__":
    main()
