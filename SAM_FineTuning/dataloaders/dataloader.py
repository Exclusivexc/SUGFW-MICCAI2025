from math import ceil
import random
import sys
from segment_anything_finetune.utils.transforms import ResizeLongestSide
import os
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import image_io
import SimpleITK as sitk

def choose_sample_list(snapshot_path: str):
    image_path_list = []
    alls = snapshot_path.split("/")
    dataset = alls[2]
    choose_method = alls[5]
    ratio = alls[6]
    
    choose_file_path = "/root/autodl-tmp/SAM_adaptive_learning/sam_data" + '/' + dataset + '/' + "iou.txt"
    with open(choose_file_path, 'r') as f:
        files = f.readlines()

    num = int(ceil(len(files) * float(ratio)))
    if choose_method == "high":
        for i in range(num):
            image_path_list.append(files[i].split(', ')[1])
    elif choose_method == "low":
        for i in range(num):
            image_path_list.append(files[len(files) - i - 1].split(', ')[1])
    
    elif choose_method == "mean":
        for i in range(num):
            image_path_list.append(files[int(len(files) // num  * i)].split(', ')[1])
    
    elif choose_method == "random":
        slices = random.sample(range(len(files)), num)
        for x in slices:
            image_path_list.append(files[x].split(', ')[1])
    elif choose_method == "uniform":
        pass

    
    
    sample_save_path = f"{os.path.abspath(snapshot_path)}/sample.txt"
    with open(sample_save_path, 'w') as f:
        for tmp in image_path_list:
            f.write(f"{tmp}\n")
        

    return image_path_list        

def get_dataset(
    snapshot_path: str,
    split="train", 
    transform=None
):
    dataset = snapshot_path.split("/")[2]
    if dataset == "Promise12":
        return Promise12(snapshot_path=snapshot_path, split=split, transform=transform)
    elif dataset == "Pancreas_MRI":
        return Pancreas_MRI(snapshot_path=snapshot_path, split=split, transform=transform)
    elif dataset == "UTAH":
        return UTAH(snapshot_path=snapshot_path, split=split, transform=transform)

class Promise12(Dataset):
    """ 
        base_dir: the location of data.
        split: the part used for training, validation or testing.
        transform: augmentation to images.
    """
    def __init__(self, snapshot_path, split="train", transform=None):
        self.base_dir = "../../sam_data/Promise12"
        self.image_path_list = []
        self.split = split
        self.transform = transform
        if split == "train":
            self.image_path_list = choose_sample_list(snapshot_path=snapshot_path)
        elif split == "all":
            self.image_path_list = [os.path.join(self.base_dir, "images", "train", file) for file in os.listdir(os.path.join(self.base_dir, "images", "train"))]
        else:     
            self.image_path_list = [os.path.join(self.base_dir, "images", split, file) for file in os.listdir(os.path.join(self.base_dir, "images", split))]
        
        
    def __len__(self):
        return len(self.image_path_list)
    

    def __getitem__(self, idx):
        image_path = self.image_path_list[idx]
        image = image_io.read_nii(image_path).astype(np.float32).squeeze()
        # image = (image - image.min()) / (image.max() - image.min()) * 255 效果没有下面的好
        image = (image - image.mean()) / image.std()
        image = np.expand_dims(image, axis=2)
        image = np.repeat(image, 3, axis=2)
        
        label = image_io.read_nii(image_path.replace("images", "labels")).squeeze().astype(np.uint8)
        label[label > 0] = 1
        
        # 加载uncertainty数据
            
        sample = {'image': image, 'label': label}
            
        if self.transform is not None:
            sample = self.transform(sample)
            
        uncertainty_path = image_path.replace("images", "uncertainty").replace(".gz", "_uncertainty.npy")
        if os.path.exists(uncertainty_path):
            uncertainty = np.load(uncertainty_path).squeeze().astype(np.float32)
            sample['uncertainty'] = uncertainty
        sample['idx'] = idx
        sample['filename'] = image_path
        
        return sample

class Pancreas_MRI(Dataset):
    """ 
        base_dir: the location of data.
        split: the part used for training, validation or testing.
        transform: augmentation to images.
    """
    def __init__(self, snapshot_path:str, split="train", transform=None):
        self.base_dir = "../../sam_data/Pancreas_MRI"
        self.image_path_list = []
        self.split = split
        self.transform = transform
        if split == "train":
            self.image_path_list = choose_sample_list(snapshot_path=snapshot_path)
        else:     
            self.image_path_list = [os.path.join(self.base_dir, "images", split, file) for file in os.listdir(os.path.join(self.base_dir, "images", split))]
        
        
    def __len__(self):
        return len(self.image_path_list)
    

    def __getitem__(self, idx):
        image_path = self.image_path_list[idx]
        image = image_io.read_nii(image_path).astype(np.float32).squeeze()
        image = (image - image.mean()) / image.std()
        image = np.expand_dims(image, axis=2)
        image = np.repeat(image, 3, axis=2)
        
        label = image_io.read_nii(image_path.replace("images", "labels")).squeeze().astype(np.uint8)
        label[label > 0] = 1
        
        # 加载uncertainty数据
        uncertainty_path = image_path.replace("images", "uncertainty")
        uncertainty = None
        if os.path.exists(uncertainty_path):
            uncertainty = image_io.read_nii(uncertainty_path).squeeze().astype(np.float32)
            
        sample = {'image': image, 'label': label}
        if uncertainty is not None:
            sample['uncertainty'] = uncertainty
            
        if self.transform is not None:
            sample = self.transform(sample)
            
        sample['idx'] = idx
        sample['filename'] = image_path
        
        return sample





class RandomCrop(object):
    """
    Crop randomly the image in a sample
    Args:
    output_size (tuple): Desired output size (height, width)
    """

    def __init__(self, output_size, with_sdf=False):
        self.output_size = output_size
        self.with_sdf = with_sdf

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        if self.with_sdf:
            sdf = sample['sdf']

        # pad the sample if necessary
        if image.shape[0] <= self.output_size[0] or image.shape[1] <= self.output_size[1]:
            pw = max((self.output_size[0] - image.shape[0]) // 2 + 3, 0)
            ph = max((self.output_size[1] - image.shape[1]) // 2 + 3, 0)
            image = np.pad(image, [(pw, pw), (ph, ph), (0, 0)], mode='constant', constant_values=0)
            label = np.pad(label, [(pw, pw), (ph, ph)], mode='constant', constant_values=0)
            if self.with_sdf:
                sdf = np.pad(sdf, [(pw, pw), (ph, ph)], mode='constant', constant_values=0)

        h, w, _ = image.shape  # h and w are height and width of the image
        w1 = np.random.randint(0, w - self.output_size[1])  # width random offset
        h1 = np.random.randint(0, h - self.output_size[0])  # height random offset

        image = image[h1:h1 + self.output_size[0], w1:w1 + self.output_size[1], :]
        label = label[h1:h1 + self.output_size[0], w1:w1 + self.output_size[1]]


        sample['image'] = image
        sample['label'] = label
        
        if self.with_sdf:
            sdf = sdf[h1:h1 + self.output_size[0], w1:w1 + self.output_size[1]]
            sample['sdf'] = sdf
        
        return sample


class RandomRotFlip(object):
    """
    Randomly rotate and flip the dataset in a sample
    """

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        
        # Random rotation (90 degrees)
        k = np.random.randint(0, 4)  # rotate 0, 90, 180, or 270 degrees
        image = np.rot90(image, k, axes=(0, 1))  # Rotate on the height and width axes
        label = np.rot90(label, k, axes=(0, 1))
        if 'uncertainty' in sample:
            sample['uncertainty'] = np.rot90(sample['uncertainty'], k, axes=(0, 1))

        # Random flip
        axis = np.random.randint(0, 2)  # 0: vertical, 1: horizontal
        image = np.flip(image, axis=axis).copy()
        label = np.flip(label, axis=axis).copy()
        if 'uncertainty' in sample:
            sample['uncertainty'] = np.flip(sample['uncertainty'], axis=axis).copy()

        sample['image'] = image
        sample['label'] = label
        return sample


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image = sample['image']
        # Ensure the image shape is (height, width, channels) and then transpose it to (channels, height, width)
        image = np.transpose(image, (2, 0, 1)).astype(np.float32)  # Convert HWC to CHW format
        
        outputs = {
            'image': torch.from_numpy(image),
            'label': torch.from_numpy(sample['label']).float()
        }
        
        if 'uncertainty' in sample:
            outputs['uncertainty'] = torch.from_numpy(sample['uncertainty']).float()
            
        if 'onehot_label' in sample:
            outputs['onehot_label'] = torch.from_numpy(sample['onehot_label']).float()
            
        return outputs

class UTAH(Dataset):
    """ 
        base_dir: the location of data.
        split: the part used for training, validation or testing.
        transform: augmentation to images.
    """
    def __init__(self, snapshot_path, split="train", transform=None):
        self.base_dir = "../../sam_data/UTAH"
        self.image_path_list = []
        self.split = split
        self.transform = transform
        if split == "train":
            self.image_path_list = choose_sample_list(snapshot_path=snapshot_path)
        elif split == "all":
            self.image_path_list = [os.path.join(self.base_dir, "images", "train", file) for file in os.listdir(os.path.join(self.base_dir, "images", "train"))]
        else:     
            self.image_path_list = [os.path.join(self.base_dir, "images", split, file) for file in os.listdir(os.path.join(self.base_dir, "images", split))]
        
        
    def __len__(self):
        return len(self.image_path_list)
    

    def __getitem__(self, idx):
        image_path = self.image_path_list[idx]
        image = image_io.read_nii(image_path).astype(np.float32).squeeze()
        # image = (image - image.min()) / (image.max() - image.min()) * 255 效果没有下面的好
        image = (image - image.mean()) / image.std()
        image = np.expand_dims(image, axis=2)
        image = np.repeat(image, 3, axis=2)
        
        label = image_io.read_nii(image_path.replace("images", "labels")).squeeze().astype(np.uint8)
        label[label > 0] = 1
        
        # 加载uncertainty数据
            
        sample = {'image': image, 'label': label}
            
        if self.transform is not None:
            sample = self.transform(sample)
            
        uncertainty_path = image_path.replace("images", "uncertainty").replace(".gz", "_uncertainty.npy")
        if os.path.exists(uncertainty_path):
            uncertainty = np.load(uncertainty_path).squeeze().astype(np.float32)
            sample['uncertainty'] = uncertainty
        sample['idx'] = idx
        sample['filename'] = image_path
        
        return sample

