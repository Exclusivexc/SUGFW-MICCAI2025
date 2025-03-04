import numpy as np
import os
from glob import glob

def resize_uncertainty(uncertainty_map):
    """
    将128x128的不确定性图缩放为64x64，通过计算每个2x2块的特定统计值
    
    Args:
        uncertainty_map: shape为(128, 128)的numpy数组
        
    Returns:
        resized_map: shape为(64, 64)的numpy数组
    """
    h, w = uncertainty_map.shape
    assert h == 128 and w == 128, f"Input shape must be (128, 128), got {uncertainty_map.shape}"
    
    # 初始化输出数组
    resized_map = np.zeros((64, 64), dtype=np.float32)
    
    # 对每个2x2的块进行处理
    for i in range(0, h, 2):
        for j in range(0, w, 2):
            block = uncertainty_map[i:i+2, j:j+2]
            
            # 计算该块的统计值：这里使用最大值作为该区域的不确定性
            # 也可以根据需求修改为其他统计方式，如平均值(mean)或最小值(min)
            block_value = np.max(block)  
            
            # 将计算结果存入对应位置
            resized_map[i//2, j//2] = block_value
            
    return resized_map

def process_folder(input_folder, output_folder):
    """
    处理文件夹中的所有.npy文件
    
    Args:
        input_folder: 输入文件夹路径，包含128x128的不确定性图
        output_folder: 输出文件夹路径，用于保存64x64的不确定性图
    """
    # 创建输出文件夹
    os.makedirs(output_folder, exist_ok=True)
    
    # 获取所有.npy文件
    npy_files = glob(os.path.join(input_folder, "*.npy"))
    
    for npy_file in npy_files:
        # 读取不确定性图
        uncertainty = np.load(npy_file)
        
        # 调整大小
        resized = resize_uncertainty(uncertainty)
        
        # 保存调整后的图
        output_path = os.path.join(output_folder, os.path.basename(npy_file))
        np.save(output_path, resized)
        
        print(f"Processed {os.path.basename(npy_file)}: {uncertainty.shape} -> {resized.shape}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Resize uncertainty maps from 128x128 to 64x64')
    parser.add_argument('--input', type=str, required=True, help='Input folder containing 128x128 uncertainty maps')
    parser.add_argument('--output', type=str, required=True, help='Output folder for 64x64 uncertainty maps')
    parser.add_argument('--method', type=str, default='max', choices=['max', 'mean', 'min'],
                        help='Method to compute block value (default: max)')
    
    args = parser.parse_args()
    
    process_folder(args.input, args.output)
