import cv2
import numpy as np
from numpy import ndarray
from typing import *



def random_crop_with_label(image: ndarray, label: ndarray, crop_height: int, crop_width: int) -> Tuple[ndarray, ndarray]:
    """
    Randomly crops a 2D image and its corresponding label, ensuring that the crop contains at least one labeled region.

    Args:
        image (numpy.ndarray): Input image as a 2D array (H, W).
        label (numpy.ndarray): Corresponding label as a 2D array (H, W).
        crop_height (int): Desired crop height.
        crop_width (int): Desired crop width.

    Returns:
        tuple: Cropped image and label as numpy arrays.
    """
    # Validate dimensions
    assert image.shape == label.shape, "Image and label must have the same dimensions."
    assert crop_height <= image.shape[0] and crop_width <= image.shape[1], "Crop size must be smaller than or equal to the image size."

    # Find the coordinates of non-zero pixels in the label
    non_zero_coords = np.column_stack(np.where(label > 0))

    # If there are no non-zero pixels, crop any part of the image
    if not non_zero_coords.size:
        max_y = image.shape[0] - crop_height
        max_x = image.shape[1] - crop_width
        start_y = np.random.randint(0, max_y + 1)
        start_x = np.random.randint(0, max_x + 1)
    else:
        # Calculate the range within which we can start cropping to ensure we crop a region with a label
        min_y = np.min(non_zero_coords[:, 0]) - crop_height
        max_y = np.max(non_zero_coords[:, 0]) + crop_height
        min_x = np.min(non_zero_coords[:, 1]) - crop_width
        max_x = np.max(non_zero_coords[:, 1]) + crop_width

        # Ensure the crop range is within the image boundaries
        min_y = max(0, min_y)
        max_y = min(image.shape[0] - crop_height, max_y)
        min_x = max(0, min_x)
        max_x = min(image.shape[1] - crop_width, max_x)

        # Randomly select the top-left corner of the crop within the valid range
        start_y = np.random.randint(min_y, max_y + 1)
        start_x = np.random.randint(min_x, max_x + 1)

    # Perform cropping
    cropped_image = image[start_y:start_y + crop_height, start_x:start_x + crop_width]
    cropped_label = label[start_y:start_y + crop_height, start_x:start_x + crop_width]

    return cropped_image, cropped_label

def random_crop(image: ndarray, label: ndarray, crop_height: int, crop_width: int) -> Tuple[ndarray, ndarray]:
    """
    Randomly crops a 2D image and its corresponding label.

    Args:
        image (numpy.ndarray): Input image as a 2D array (H, W).
        label (numpy.ndarray): Corresponding label as a 2D array (H, W).
        crop_height (int): Desired crop height.
        crop_width (int): Desired crop width.

    Returns:
        tuple: Cropped image and label as numpy arrays.
    """
    # Validate dimensions
    assert image.shape == label.shape, "Image and label must have the same dimensions."
    assert crop_height <= image.shape[0] and crop_width <= image.shape[1], \
        "Crop size must be smaller than or equal to the image size."

    # Randomly select the top-left corner of the crop
    max_y = image.shape[0] - crop_height
    max_x = image.shape[1] - crop_width
    start_y = np.random.randint(0, max_y + 1)
    start_x = np.random.randint(0, max_x + 1)

    # Perform cropping
    cropped_image = image[start_y:start_y + crop_height, start_x:start_x + crop_width]
    cropped_label = label[start_y:start_y + crop_height, start_x:start_x + crop_width]

    return cropped_image, cropped_label

def pad_image_and_label(image: ndarray, label: ndarray, width: int, height: int):
    """
    Pad the image and label to the specified width and height.

    Parameters:
    - image: 2D NumPy array representing the image.
    - label: 2D NumPy array representing the label.
    - width: The desired width after padding.
    - height: The desired height after padding.

    Returns:
    - padded_image: The padded image.
    - padded_label: The padded label.
    """
    # 计算需要填充的宽度和高度差值
    pad_width = width - image.shape[1] if width > image.shape[1] else 0
    pad_height = height - image.shape[0] if height > image.shape[0] else 0

    # 计算左右填充的宽度
    pad_left = pad_width // 2
    pad_right = pad_width - pad_left

    # 计算上下填充的高度
    pad_top = pad_height // 2
    pad_bottom = pad_height - pad_top

    # 使用np.pad进行填充操作
    padded_image = np.pad(image, ((pad_top, pad_bottom), (pad_left, pad_right)), mode='constant', constant_values=0)
    padded_label = np.pad(label, ((pad_top, pad_bottom), (pad_left, pad_right)), mode='constant', constant_values=0)

    return padded_image, padded_label



    
if __name__ == "__main__":
    image = np.random.random((224, 224))
    label = image
    image, label = pad_image_and_label(image, label, 500, 500)
    print(image.shape)