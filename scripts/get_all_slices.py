import os
import nibabel as nib
import numpy as np
from collections import defaultdict
import re
from tqdm import tqdm
import shutil

def get_case_slice_number(filename):
    """从文件名中提取case编号和slice编号"""
    match = re.match(r'Case(\d+)_(\d+)\.nii\.gz', filename)
    if match:
        return int(match.group(1)), int(match.group(2))
    return None, None

def process_dataset(image_dir, label_dir, output_image_dir, output_label_dir):
    """处理数据集，只保留label最大值为1的切片"""
    os.makedirs(output_image_dir, exist_ok=True)
    os.makedirs(output_label_dir, exist_ok=True)
    
    # 按case分组所有文件
    case_dict = defaultdict(list)
    for file_name in os.listdir(label_dir):
        if file_name.endswith('.nii.gz'):
            case_num, slice_num = get_case_slice_number(file_name)
            if case_num is not None:
                case_dict[case_num].append((slice_num, file_name))
    
    # 处理每个case
    for case_num in tqdm(sorted(case_dict.keys()), desc="Processing cases"):
        # 获取该case的所有切片信息
        valid_slices = set()
        
        for slice_num, file_name in sorted(case_dict[case_num]):
            label_path = os.path.join(label_dir, file_name)
            label_nii = nib.load(label_path)
            label_data = label_nii.get_fdata()
            
            max_label = np.max(label_data)
            if max_label == 1:  # 只保留label最大值为1的切片
                # 复制图像
                src_image = os.path.join(image_dir, file_name)
                dst_image = os.path.join(output_image_dir, file_name)
                if os.path.exists(src_image):
                    shutil.copy2(src_image, dst_image)
                
                # 复制标签
                src_label = os.path.join(label_dir, file_name)
                dst_label = os.path.join(output_label_dir, file_name)
                if os.path.exists(src_label):
                    shutil.copy2(src_label, dst_label)

def main():
    base_dir = "/media/ubuntu/maxiaochuan/SAM_adaptive_learning/sam_data/Promise12"
    
    # 处理训练集、验证集和测试集
    for subset in ['train', 'valid', 'test']:
        print(f"\nProcessing {subset} set...")
        
        image_dir = os.path.join(base_dir, 'images', subset)
        label_dir = os.path.join(base_dir, 'labels', subset)
        
        output_image_dir = os.path.join(base_dir, 'images_processed', subset)
        output_label_dir = os.path.join(base_dir, 'labels_processed', subset)
        
        if not os.path.exists(image_dir) or not os.path.exists(label_dir):
            print(f"Skipping {subset} - directories not found")
            continue
            
        process_dataset(
            image_dir=image_dir,
            label_dir=label_dir,
            output_image_dir=output_image_dir,
            output_label_dir=output_label_dir
        )

if __name__ == "__main__":
    main()
