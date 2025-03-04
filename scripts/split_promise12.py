import os
import shutil
import SimpleITK as sitk
from PIL import Image
import numpy as np
from tqdm import tqdm

def get_mhd_data():
    # 定义目录
    base_dir = '/media/ubuntu/maxiaochuan/SAM_adaptive_learning/data/Promise12'
    training_data_dir = os.path.join(base_dir, 'training_data')
    test_data_dir = os.path.join(base_dir, 'test_data')
    save_dir = "/media/ubuntu/maxiaochuan/SAM_adaptive_learning/raw_data/Promise12"
    labels_dir = os.path.join(save_dir, 'labels')
    images_dir = os.path.join(save_dir, 'images')

    # 创建labels和images目录，如果不存在的话
    os.makedirs(labels_dir, exist_ok=True)
    os.makedirs(images_dir, exist_ok=True)

    # 用于追踪重命名的文件计数
    image_count = 1

    # 处理training_data和test_data中的文件
    for data_dir in [training_data_dir, test_data_dir]:
        for filename in tqdm(os.listdir(data_dir)):
            file_path = os.path.join(data_dir, filename)
            
            # 只处理.mhd文件
            if filename.endswith('.mhd'):
                if filename.endswith('segmentation.mhd'):
                    # print(filename)
                    # 处理segmentation.mhd文件，移动到labels并重命名
                    label = sitk.GetArrayFromImage(sitk.ReadImage(file_path))
                    image = sitk.GetArrayFromImage(sitk.ReadImage(file_path.replace("_segmentation", "")))
                    for slice_index in range(image.shape[0]):  # size[2]是z轴切片数
                        slice_image = image[slice_index, :, :]
                        slice_label = label[slice_index, :, :]
                        slice_image = np.expand_dims(slice_image, axis=0)
                        slice_label = np.expand_dims(slice_label, axis=0)
                        # 构建新文件名
                        slice_image = (slice_image - slice_image.min()) / (slice_image.max() - slice_image.min())
                        new_filename = f'case{image_count:02d}_{slice_index+1:02d}.nii.gz'
                        new_image_path = os.path.join(save_dir, "images", new_filename)
                        new_label_path = os.path.join(save_dir, "labels", new_filename)
                        # 保存切片
                        sitk.WriteImage(sitk.GetImageFromArray(slice_image), new_image_path)
                        sitk.WriteImage(sitk.GetImageFromArray(slice_label), new_label_path)

                    image_count += 1



get_mhd_data()             

# sitk.ReadImage("/media/ubuntu/maxiaochuan/SAM_adaptive_learning/data/Promise12/training_data/Case11_segmentation.mhd")