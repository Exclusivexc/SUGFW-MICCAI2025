import numpy as np
import os
from glob import glob
from tqdm import tqdm

def resize_uncertainty(uncertainty_map):
    """
    将128x128的不确定性图缩放为64x64，使用平均值聚合2x2块
    """
    h, w = uncertainty_map.shape
    assert h == 128 and w == 128, f"Input shape must be (128, 128), got {uncertainty_map.shape}"
    
    resized_map = np.zeros((64, 64), dtype=np.float32)
    
    # 对每个2x2的块计算平均值而不是最大值
    for i in range(0, h, 2):
        for j in range(0, w, 2):
            block = uncertainty_map[i:i+2, j:j+2]
            resized_map[i//2, j//2] = np.mean(block)  # 使用mean替代max
            
    return resized_map

def process_directory(input_dir, output_dir):
    """处理单个目录中的所有npy文件"""
    os.makedirs(output_dir, exist_ok=True)
    npy_files = glob(os.path.join(input_dir, "*.npy"))
    
    for npy_file in tqdm(npy_files, desc=f"Processing {os.path.basename(input_dir)}"):
        # 读取并处理文件
        uncertainty = np.load(npy_file)
        resized = resize_uncertainty(uncertainty)
        
        # 保存处理后的文件
        output_path = os.path.join(output_dir, os.path.basename(npy_file))
        np.save(output_path, resized)

def main():
    # 基础目录
    base_dir = "/media/ubuntu/maxiaochuan/SAM_adaptive_learning/sam_data/UTAH/uncertainty"
    output_base = "/media/ubuntu/maxiaochuan/SAM_adaptive_learning/sam_data/UTAH/uncertainty_64"
    
    # 要处理的子目录
    subdirs = ["train", "valid", "test"]
    
    # 处理每个子目录
    for subdir in subdirs:
        input_dir = os.path.join(base_dir, subdir)
        output_dir = os.path.join(output_base, subdir)
        print(f"\nProcessing directory: {input_dir}")
        
        if not os.path.exists(input_dir):
            print(f"Warning: Directory {input_dir} does not exist!")
            continue
            
        process_directory(input_dir, output_dir)
        print(f"Completed processing {subdir}")

if __name__ == "__main__":
    main()
