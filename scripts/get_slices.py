import os
import SimpleITK as sitk
from tqdm import tqdm




def extract_slices_and_save(src_folder, dst_folder):
    # 如果目标文件夹不存在，则创建它
    os.makedirs(dst_folder, exist_ok=True)
    # 遍历源文件夹中的所有文件
    for file_name in tqdm(os.listdir(src_folder)):
        file_path = os.path.join(src_folder, file_name)

        # 跳过目录
        if os.path.isdir(file_path):
            continue

        # 检查是否为.nii.gz文件
        if file_name.endswith('.nii.gz'):
            # 读取NIfTI图像
            image = sitk.ReadImage(file_path)
            
            # 获取图像的尺寸（如：d, w, h）
            size = image.GetSize()
            num_slices = size[2]  # 假设是3D图像，z轴代表切片数
            
            # 遍历每个切片
            t = file_name.replace(".nii.gz", "")
            print(t)
            for i in range(num_slices):
                # 提取第i个切片（假设z轴是切片方向）
                slice_image = image[:, :, i]

                # 生成新的文件名
                new_file_name = f"{t}_{i:02d}.nii.gz"
                new_file_path = os.path.join(dst_folder, new_file_name)

                # 保存切片为新的.nii.gz文件
                sitk.WriteImage(slice_image, new_file_path)
                print(f"已保存切片：{new_file_name}")
                
image_dir = "/media/ubuntu/maxiaochuan/SAM_adaptive_learning/raw_data/Promise12/images/train"
label_dir = "/media/ubuntu/maxiaochuan/SAM_adaptive_learning/raw_data/Promise12/labels/train"

out_image_dir = "/media/ubuntu/maxiaochuan/SAM_adaptive_learning/processed_data/Promise12/images/validation"
out_label_dir = "/media/ubuntu/maxiaochuan/SAM_adaptive_learning/processed_data/Promise12/labels/validation"
extract_slices_and_save(image_dir, out_image_dir)
extract_slices_and_save(label_dir, out_label_dir)