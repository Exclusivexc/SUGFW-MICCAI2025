from typing import * 
import numpy as np
from numpy import ndarray
import cv2
from skimage import exposure
from scipy import ndimage
import random

""" linear transform """

def gray_inversion(image: ndarray) -> ndarray:
    return 255 - image

    
def contrast_stretching(image: ndarray, O_min: int=0, O_max: int=255) -> ndarray:
    """ 
    effect: Suitable for images with a narrow brightness range, improving overall contrast by simple stretching
    """
    image = (image - image.min()) / (image.max() - image.min()) * (O_max - O_min) + O_min
    image = np.clip(image, 0, 255).astype(np.uint8)
    return image

def contrast_and_bightness_enhancing(image: ndarray, alpha: float=2.0, beta: float=30) -> ndarray:
    """ 
    effect: The grayscale values of an image are linearly enlarged or compressed by multiplying them by a contrast factor alpha.
    formula: O(x, y) = alpha * I(x, y) + beta 
    alpha > 1: increase contrast; 0 < alpha < 1: reduce contrast.
    beta: increase or reduce intensity with a threshold of 0.
    """
    image = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
    # image = image * alpha + beta 
    return image 
    
""" Nonlinear transform """
def exponential_transformation(image: ndarray, c: int=255) -> ndarray:
    """
    effect: Enhances the darker or lighter parts of an image.
    formula: O(x, y) = c * (e^I(x, y) - 1)
    I(x, y) is in [0, 255]
    e ≈ 2.71828
    c is a constant to control the output intensity.
        (1) larger c results in a brighter image.
        (2) lower c results in weak augmentation, suitable for sightly adjustment
    """
    image = image / 255.0
    exp_image = c * (np.exp(image) - 1)
    exp_image = np.clip(exp_image, 0, 255).astype(np.uint8)
    return exp_image

def gamma_transformation(image: ndarray, gamma: float=1.0) -> ndarray:
    """ 
    effect: Adjust the image brightness by controlling the gamma value γ
    formula: O(x, y) = c * I(x, y) ^ gamma
    I(x, y) is normalized to [0, 1]
    c is a constant in [0, 255]
    gamma:
        (1) 0 < gamma < 1: brighten the dark area, suitable for darker images.
        (2) gamma > 1: darken the bright area, sutable for overexposed images.
    commonly used gamma:
        (1) gamma = 0.4~0.7
        (2) gamma = 1.5~2.5
    """
    inv_gamma = 1.0 / gamma 
    gamma_table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(image, gamma_table)

def logarithmic_transformation(image: ndarray, c: int=1) -> ndarray:
    """ 
    effect: Compress highlight areas and enhance dark details
    formula: O(x, y) = c * log(1 + I(x, y))
    if the pixel value is in [0, 255], c = 255 / log(256) ≈ 45.98 
    """
    log_image = c * (np.log1p(image.astype(np.float32)))  # log1p 是 log(1 + x)
    log_image = np.clip(log_image, 0, 255).astype(np.uint8)
    return log_image

def random_noise(image, mean=0.0, var=1):
    gaussian_noise = np.random.normal(mean, var, (image.shape[0], image.shape[1]))
    image = image + gaussian_noise
    return image

def random_rescale_intensity(image):
    image = exposure.rescale_intensity(image)
    return image


def laplacian_sharpening(image, alpha=1):
    # 使用拉普拉斯算子来检测边缘
    laplacian = ndimage.laplace(image)
    
    # 锐化操作：原图像 + alpha * 边缘增强部分
    sharpened_image = image + alpha * laplacian
    return sharpened_image

def random_sharpening(image):
    blurred = ndimage.gaussian_filter(image, 3)
    blurred_filter = ndimage.gaussian_filter(blurred, 1)
    alpha = random.randrange(1, 10)
    image = blurred + alpha * (blurred - blurred_filter)
    return image

def histogram_equalization(image: ndarray) -> ndarray:
    """ 
    effect: Automatically adjust the contrast of the image, suitable for images with uneven brightness distribution
    """
    image = cv2.equalizeHist(image)
    return image

def contrast_limited_adaptive_histogram_equalization(image: ndarray, cliplimit: float=2.0, tileGridSize: Tuple[int, int]=(8, 8)) -> ndarray:
    """ 
    CLAHE divides the image into small blocks (called "tiles") and performs histogram equalization on each tile.
    After local equalization, CLAHE merges the results of different blocks through interpolation techniques to avoid obvious boundaries and discontinuities.
        (1): clipLimit: The threshold used to limit the contrast. The smaller the value, the less contrast enhancement, and the larger the value, the more contrast enhancement.
        (2): tileGridSize: Specifies the size of the divided tiles (for example, (8, 8) means dividing the image into 8x8 tiles), which affects the effect of local equalization.
    """

    clahe = cv2.createCLAHE(clipLimit=cliplimit, tileGridSize=tileGridSize)
    clahe_image = clahe.apply(image)
    
    return clahe_image