import numpy as np
from numpy import ndarray
from PIL import Image
import SimpleITK as sitk


def read_png(image_path: str) -> ndarray:
    return np.array(Image.open(image_path))

    
    
def save_png(image: ndarray, save_path: str) -> None:
    img = Image.fromarray(image)
    img.save(save_path)
    return 

def read_nii(image_path: str) -> ndarray:
    return sitk.GetArrayFromImage(sitk.ReadImage(image_path))

def save_nii(image: ndarray, save_path: str) -> None:
    sitk.WriteImage(sitk.GetImageFromArray(image), save_path)
    return 