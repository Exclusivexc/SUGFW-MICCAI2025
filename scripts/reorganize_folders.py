import os
import shutil
from pathlib import Path

def reorganize_folders(base_dir: str):
    """
    重组文件夹结构
    从:
    train/
        images/
        labels/
        uncertainty/
    valid/
        images/
        labels/
        uncertainty/
    test/
        images/
        labels/
        uncertainty/
        
    到:
    images/
        train/
        valid/
        test/
    labels/
        train/
        valid/
        test/
    uncertainty/
        train/
        valid/
        test/
    """
    # 创建新的目录结构
    new_structure = {
        'images': ['train', 'valid', 'test'],
        'labels': ['train', 'valid', 'test'],
        'uncertainty': ['train', 'valid', 'test']
    }
    
    # 创建新目录
    for main_folder, subfolders in new_structure.items():
        for subfolder in subfolders:
            new_path = os.path.join(base_dir, main_folder, subfolder)
            os.makedirs(new_path, exist_ok=True)
    
    # 移动文件
    for split in ['train', 'valid', 'test']:
        for data_type in ['images', 'labels', 'uncertainty']:
            # 源目录
            src_dir = os.path.join(base_dir, split, data_type)
            # 目标目录
            dst_dir = os.path.join(base_dir, data_type, split)
            
            if os.path.exists(src_dir):
                # 移动所有文件
                for file_name in os.listdir(src_dir):
                    src_file = os.path.join(src_dir, file_name)
                    dst_file = os.path.join(dst_dir, file_name)
                    shutil.move(src_file, dst_file)
                    print(f"Moved: {src_file} -> {dst_file}")
                
                # 删除空的源目录
                os.rmdir(src_dir)
                print(f"Removed empty directory: {src_dir}")
        
        # 删除空的split目录
        split_dir = os.path.join(base_dir, split)
        if os.path.exists(split_dir) and len(os.listdir(split_dir)) == 0:
            os.rmdir(split_dir)
            print(f"Removed empty directory: {split_dir}")

def main():
    # 设置基础目录路径
    base_dir = "/media/ubuntu/maxiaochuan/SAM_adaptive_learning/sam_data/Promise12"  # 根据实际路径修改
    
    # 执行重组
    print("Starting folder reorganization...")
    reorganize_folders(base_dir)
    print("Folder reorganization completed!")

if __name__ == "__main__":
    main()
