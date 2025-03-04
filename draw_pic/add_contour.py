import SimpleITK as sitk
import os
from PIL import Image, ImageEnhance
import numpy as np
import cv2

def add_contour(In, Seg, Color=(255, 0, 0)):
    """
    add segmentation contour to an input image

    In: Input PIL.Image object, should be an RGB image
    Seg: segmentation mask represented by a PIL.Image object
    Color: a vector specifying the color of contour
    Out: output PIL.Image object with segmentation contour overlayed
    """
    Out = In.convert("RGB").copy()  # Ensure the image is in RGB mode
    [H, W] = In.size
    for i in range(H):
        for j in range(W):
            if(i==0 or i==H-1 or j==0 or j == W-1):
                if(Seg.getpixel((i,j))!=0):
                    Out.putpixel((i,j), Color)
            elif(Seg.getpixel((i,j))!=0 and  \
                 not(Seg.getpixel((i-1,j))!=0 and \
                     Seg.getpixel((i+1,j))!=0 and \
                     Seg.getpixel((i,j-1))!=0 and \
                     Seg.getpixel((i,j+1))!=0)):
                     Out.putpixel((i,j), Color)
    return Out

def normalize_array(array):
    """Normalize array to 0-255 range"""
    array = array.astype(float)
    array = ((array - array.min()) / (array.max() - array.min()) * 255).astype(np.uint8)
    return array

def get_bbox_with_padding(mask, padding_ratio=1):
    """获取mask的边界框，并添加padding"""
    # 找到非零像素的坐标
    y, x = np.nonzero(np.array(mask))
    if len(x) == 0 or len(y) == 0:
        return None
    
    # 计算边界框
    x_min, x_max = np.min(x), np.max(x)
    y_min, y_max = np.min(y), np.max(y)
    
    # 计算padding大小
    width = x_max - x_min
    height = y_max - y_min
    padding_x = int(width * padding_ratio)
    padding_y = int(height * padding_ratio)
    
    # 添加padding并确保不超出图像边界
    x_min = max(0, x_min - padding_x)
    x_max = min(mask.size[0], x_max + padding_x)
    y_min = max(0, y_min - padding_y)
    y_max = min(mask.size[1], y_max + padding_y)
    
    return (x_min, y_min, x_max, y_max)

def crop_and_resize(image, bbox, original_size):
    """裁剪图像并调整回原始大小"""
    # 裁剪
    cropped = image.crop(bbox)
    # 调整回原始大小
    resized = cropped.resize(original_size, Image.Resampling.LANCZOS)
    return resized

def enhance_contrast(image):
    """增强图像对比度"""
    # 如果输入是PIL Image，转换为numpy数组
    if isinstance(image, Image.Image):
        image_array = np.array(image)
    else:
        image_array = image
    
    # 如果是RGB图像，分别对每个通道进行直方图均衡化
    if len(image_array.shape) == 3:
        enhanced = np.zeros_like(image_array)
        for i in range(3):
            enhanced[:,:,i] = cv2.equalizeHist(image_array[:,:,i])
    else:
        enhanced = cv2.equalizeHist(image_array)
    
    # 进一步增强对比度
    enhanced_image = Image.fromarray(enhanced)
    # enhancer = ImageEnhance.Contrast(enhanced_image)
    # enhanced_image = enhancer.enhance(1.5)  # 增强系数可调
    
    # 增加亮度
    enhancer = ImageEnhance.Brightness(enhanced_image)
    enhanced_image = enhancer.enhance(0.5)  # 亮度系数可调
    
    return enhanced_image

input_dir = "/root/autodl-tmp/SAM_adaptive_learning/draw_pic/U1"
output_dir = "/root/autodl-tmp/SAM_adaptive_learning/draw_pic/U1/output"
os.makedirs(output_dir, exist_ok=True)

# Read the main image
image_path = os.path.join(input_dir, "image.nii.gz")
image = sitk.ReadImage(image_path)
image_array = sitk.GetArrayFromImage(image)
normalized_image = normalize_array(image_array).squeeze()
normalized_image = np.stack([normalized_image] * 3, axis=-1)
image_pil = Image.fromarray(normalized_image).convert("RGB")

# 增强图像对比度
image_pil = enhance_contrast(image_pil)

# Read the label image
label_path = os.path.join(input_dir, "label.nii.gz")
label = sitk.ReadImage(label_path)
label_array = sitk.GetArrayFromImage(label).squeeze()
label_pil = Image.fromarray(label_array).convert("L")

# Process all PNG files in the directory
for file in os.listdir(input_dir):
    if file.endswith(".png"):
        mask_path = os.path.join(input_dir, file)
        mask = Image.open(mask_path).convert("L")
        
        # 添加轮廓
        out_image = add_contour(image_pil, mask)
        out_image = add_contour(out_image, label_pil, Color=(0, 255, 0))
        
        # 获取groundtruth的边界框
        bbox = get_bbox_with_padding(label_pil)
        if bbox is not None:
            # 裁剪并放大
            out_image = crop_and_resize(out_image, bbox, out_image.size)
        
        # 保存结果
        save_path = os.path.join(output_dir, file)
        out_image.save(save_path)
        
        # 可选：保存裁剪区域的可视化结果
        if bbox is not None:
            debug_image = out_image.copy()
            from PIL import ImageDraw
            draw = ImageDraw.Draw(debug_image)
            draw.rectangle(bbox, outline=(255, 255, 0), width=2)
            debug_save_path = os.path.join(output_dir, f"debug_{file}")
            debug_image.save(debug_save_path)
