import os
import numpy as np
import shutil
import SimpleITK as sitk

def load_prostate_ISBI_2013(img_dir="/media/ubuntu/maxiaochuan/SAM_adaptive_learning/data/Prostate ISBI2013"):

    # 义源目录
    iter_dir = ['Prostate-3T', 'PROSTATE-DIAGNOSIS']
    for base_dir in iter_dir: 
        base_dir = os.path.join(img_dir, base_dir)
        # 遍历每个患者文件夹
        for patient_folder in os.listdir(base_dir):
            patient_path = os.path.join(base_dir, patient_folder)
            
            if os.path.isdir(patient_path):
                # 遍历每个子目录
                for root, dirs, files in os.walk(patient_path):
                    for file in files:
                        if file.endswith('.dcm'):
                            # 构建源文件路径
                            file_path = os.path.join(root, file)
                            # 找到目标路径
                            target_path = os.path.join(patient_path, file)
                            # 移动文件到目标路径
                            shutil.move(file_path, target_path)

                # 删除空的中间文件夹
                for root, dirs, files in os.walk(patient_path, topdown=False):
                    for dir in dirs:
                        dir_path = os.path.join(root, dir)
                        # 删除空文件夹
                        if not os.listdir(dir_path):
                            os.rmdir(dir_path)

def load(label_path=r"/media/ubuntu/maxiaochuan/SAM_adaptive_learning/raw_data/Prostate ISBI2013/labels"):
    alls = os.listdir(label_path)
    for i in alls:
        pp = i.split(".")[0]
        os.makedirs(os.path.join(label_path, pp), exist_ok=True)
        label = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(label_path, i)))
        for j in range(label.shape[0]):
            tt = label[j]
            sitk.WriteImage(sitk.GetImageFromArray(tt), os.path.join(label_path, pp, f"1-{j + 1:02d}.dcm"))


# load_prostate_ISBI_2013()
def delete(label_path=r"/media/ubuntu/maxiaochuan/SAM_adaptive_learning/raw_data/Prostate ISBI2013/labels"):
    t = os.listdir(label_path)
    for i in t:
        p = os.path.join(label_path, i)
        if p.endswith(".nrrd"):
            print(p)
            os.remove(p)
            
def folder_to_flatten(base_f="/media/ubuntu/maxiaochuan/SAM_adaptive_learning/raw_data/Prostate_ISBI2013"):

    # 定义源目录
    phase = ["images", "labels"]
    for p in phase:
        dd = os.listdir(os.path.join(base_f, p))
        for d in dd:
            if os.path.isdir(d):
                os.rmdir(d)
        # for d in dd:
        #     base_dir = os.path.join(base_f, p, d)
            
        #     # 遍历 b 目录
        #     if os.path.isdir(base_dir):
        #         print(base_dir)
        #         os.rmdir(base_dir)
                # for subfolder in os.listdir(base_dir):
                #     subfolder_path = os.path.join(base_dir, subfolder)
                    
                #     # 只处理目录
                #     # 构造新的目录名
                #     new_folder_name = f"{d}-{subfolder}"
                #     new_folder_path = os.path.join(f'/media/ubuntu/maxiaochuan/SAM_adaptive_learning/raw_data/Prostate_ISBI2013/{p}', new_folder_name)
                #     # 重命名目录
                #     # print(subfolder_path, new_folder_path)
                #     os.rename(subfolder_path, new_folder_path)

    print("目录结构已更新。")

# folder_to_flatten()