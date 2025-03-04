#%% preprocess dataset
import numpy as np
import os
import SimpleITK as sitk
from skimage import exposure
import shutil

from tqdm import tqdm

def clip_intensity(image, percent=0.99):
    cdf = exposure.cumulative_distribution(image)
    watershed = cdf[1][cdf[0] >= percent][0]
    return np.clip(image, image.min(), watershed)


if __name__ == "__main__":
    dir_root = '/media/ubuntu/maxiaochuan/SAM_adaptive_learning/data/Promise12'
    sets = ['training_data', 'test_data', 'livechallenge_test_data']

    for set in sets:
        dir = f'/media/ubuntu/maxiaochuan/SAM_adaptive_learning/data/Promise_nii/{set}/images'
        dir_label = f'/media/ubuntu/maxiaochuan/SAM_adaptive_learning/data/Promise_nii/{set}/labels'
        dir_save_image = f'/media/ubuntu/maxiaochuan/SAM_adaptive_learning/data/Promise_nii_preprocessed/{set}/images'
        dir_save_label = f'/media/ubuntu/maxiaochuan/SAM_adaptive_learning/data/Promise_nii_preprocessed/{set}/labels'
        os.makedirs(dir_save_image, exist_ok=True)
        os.makedirs(dir_save_label, exist_ok=True)

        filenames = sorted(os.listdir(dir), reverse=False)
        for filename in tqdm(filenames):
            image_itk = sitk.ReadImage(f'{dir}/{filename}')
            spacing = image_itk.GetSpacing()
            image_volume = sitk.GetArrayFromImage(image_itk).squeeze()
            image_volume_preprocessed = clip_intensity(image_volume, percent=0.99).astype(np.int16)
            image_volume_preprocessed_itk = sitk.GetImageFromArray(image_volume_preprocessed)
            image_volume_preprocessed_itk.CopyInformation(image_itk)
            sitk.WriteImage(image_volume_preprocessed_itk, f'{dir_save_image}/{filename}')
            shutil.copy(f'{dir_label}/{filename}', f'{dir_save_label}/{filename}')