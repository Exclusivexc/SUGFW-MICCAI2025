import numpy as np
import os
import argparse

def read_npy_shape(file_path):
    """
    读取 NPY 文件并打印其形状
    
    Args:
        file_path (str): NPY 文件的路径
    """
    try:
        # 读取 NPY 文件
        data = np.load(file_path)
        
        # 打印文件名和形状
        print(f"File: {os.path.basename(file_path)}")
        print(f"Shape: {data.shape}")
        print(f"Data type: {data.dtype}")
        
    except Exception as e:
        print(f"Error reading file {file_path}: {str(e)}")

def main():
    # 创建命令行参数解析器
    parser = argparse.ArgumentParser(description='Read and display NPY file shape')
    parser.add_argument("file_path", default='/media/ubuntu/maxiaochuan/SAM_adaptive_learning/sam_data/UTAH/uncertainty/train/Case_99_81.nii_uncertainty.npy', help='Path to the NPY file')
    args = parser.parse_args()
    
    read_npy_shape(args.file_path)

if __name__ == "__main__":
    main()