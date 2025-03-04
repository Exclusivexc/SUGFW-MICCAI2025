import numpy as np
from numpy import ndarray
from typing import *
def Zscore_Norm(image: ndarray) -> ndarray: 
    image = (image - image.mean()) / image.std()
    return image

def Min_Max_Norm(image: ndarray) -> ndarray:
    image = (image - image.min()) / (image.max() - image.min())
    return image


def Range_Norm(image: ndarray, target_range: List[int]=[-1, 1]) -> ndarray:
    normalized_image = (image - image.min()) / (image.max() - image.min()) * (target_range[1] - target_range[0]) + target_range[0]
    
    return normalized_image