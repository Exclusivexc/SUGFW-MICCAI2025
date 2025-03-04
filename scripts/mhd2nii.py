import SimpleITK as sitk
import os

from tqdm import tqdm 

def mhd2nii(mhd_folder, save_folder):
    phase = os.listdir(mhd_folder)    
    os.makedirs(save_folder, exist_ok=True)
    for p in phase:
        base_dir = os.path.join(mhd_folder, p)
        save_dir = os.path.join(save_folder, p)
        os.makedirs(save_dir, exist_ok=True)
        files = os.listdir(base_dir)
        for file in tqdm(files):
            if file.endswith(".mhd"):
                t = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(base_dir, file)))
                sitk.WriteImage(sitk.GetImageFromArray(t), os.path.join(save_dir, file.replace(".mhd", ".nii.gz").replace("_segmentation", "")))
            
if __name__ == "__main__":
    mhd_folder = "/media/ubuntu/maxiaochuan/SAM_adaptive_learning/data/Promise12"
    save_folder = "/media/ubuntu/maxiaochuan/SAM_adaptive_learning/data/Promise_nii"
    phase = ['training_data', 'test_data', 'livechallenge_test_data']
    for p in phase:
        mhd2nii(os.path.join(mhd_folder, p), os.path.join(save_folder, p))
