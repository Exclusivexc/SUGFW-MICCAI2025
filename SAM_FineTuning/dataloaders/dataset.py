import os
import numpy as np
from torch.utils.data import Dataset
import SimpleITK as sitk
import torchvision.transforms as transforms
import torch


class Prostate_ISBI2013(Dataset):
    """ 
        base_dir: the location of data.
        split: the part used for training, validation or testing.
        transform: augmentation to images.
    """
    def __init__(self, base_dir="/media/ubuntu/maxiaochuan/SAM_adaptive_learning/processed_data/Prostate_ISBI2013", split="Fully", transform=None):
        self.base_dir = base_dir
        self.sample_list = []
        self.split = split
        self.transform = transform
        self.sample_list = os.listdir(os.path.join(self.base_dir, "images", split))
        
        
    def __len__(self):
        return len(self.sample_list)
    

    def __getitem__(self, idx):
        filename = self.sample_list[idx]
        file_path = os.path.join(self.base_dir, "images", self.split, filename)
        image = sitk.GetArrayFromImage(sitk.ReadImage(file_path)).astype(np.float32)
        m = image.mean()
        s = image.std()
        image = (image - m) / s

        label = sitk.GetArrayFromImage(sitk.ReadImage(file_path.replace("images", "labels"))).astype(np.float32)
        label[label > 0] = 1
        label /= 1.0

        
        image = np.resize(image, (240, 240))
        label = np.resize(label, (240, 240))
        sample = {'image': image, 'label': label}
        if self.transform is not None:
            sample = self.transform(sample)
        sample['idx'] = idx
        sample['filename'] = filename
        
        return sample
    
class Pancreas_MRI(Dataset):
    def __init__(self):
        pass

    def __len__(self):
        pass
    
    def __getitem__(self, index):
        pass
    
class BraTS2020(Dataset):
    def __init__(self):
        pass

    def __len__(self):
        pass
    
    def __getitem__(self, index):
        pass
    

class Promise12(Dataset):
    """ 
        base_dir: the location of data.
        split: the part used for training, validation or testing.
        transform: augmentation to images.
    """
    def __init__(self, base_dir="/media/ubuntu/maxiaochuan/SAM_adaptive_learning/processed_data/Promise12", split="train", transform=None):
        self.base_dir = base_dir
        self.sample_list = []
        self.split = split
        self.transform = transform
        self.sample_list = os.listdir(os.path.join(self.base_dir, "images", split))
        
        
    def __len__(self):
        return len(self.sample_list)
    

    def __getitem__(self, idx):
        filename = self.sample_list[idx]
        file_path = os.path.join(self.base_dir, "images", self.split, filename)
        image = sitk.GetArrayFromImage(sitk.ReadImage(file_path)).astype(np.float32).squeeze()
        image = (image - image.mean()) / image.std()
        label = sitk.GetArrayFromImage(sitk.ReadImage(file_path.replace("images", "labels"))).squeeze().astype(np.uint8)
        label[label > 0] = 1
        # print(image.shape, label.shape)
        # image = np.resize(image, (400, 400))
        # label = np.resize(label, (400, 400))
        sample = {'image': image, 'label': label}
        if self.transform is not None:
            sample = self.transform(sample)
        sample['idx'] = idx
        sample['filename'] = file_path
        
        return sample

    

class RandomCrop(object):
    """
    Crop randomly the image in a sample
    Args:
    output_size (int): Desired output size
    """

    def __init__(self, output_size, with_sdf=False):
        self.output_size = output_size
        self.with_sdf = with_sdf

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        if self.with_sdf:
            sdf = sample['sdf']

        # pad the sample if necessary
        if label.shape[0] <= self.output_size[0] or label.shape[1] <= self.output_size[1]:
            pw = max((self.output_size[0] - label.shape[0]) // 2 + 3, 0)
            ph = max((self.output_size[1] - label.shape[1]) // 2 + 3, 0)
            image = np.pad(image, [(pw, pw), (ph, ph)], mode='constant', constant_values=0)
            label = np.pad(label, [(pw, pw), (ph, ph)], mode='constant', constant_values=0)
            if self.with_sdf:
                sdf = np.pad(sdf, [(pw, pw), (ph, ph)], mode='constant', constant_values=0)

        (w, h) = image.shape
        w1 = np.random.randint(0, w - self.output_size[0])
        h1 = np.random.randint(0, h - self.output_size[1])

        label = label[w1:w1 + self.output_size[0], h1:h1 + self.output_size[1]]
        image = image[w1:w1 + self.output_size[0], h1:h1 + self.output_size[1]]
        if self.with_sdf:
            sdf = sdf[w1:w1 + self.output_size[0], h1:h1 + self.output_size[1]]
            return {'image': image, 'label': label, 'sdf': sdf}
        else:
            return {'image': image, 'label': label}


class RandomRotFlip(object):
    """
    Crop randomly flip the dataset in a sample
    Args:
    output_size (int): Desired output size
    """

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        k = np.random.randint(0, 4)
        image = np.rot90(image, k)
        label = np.rot90(label, k)
        axis = np.random.randint(0, 2)
        image = np.flip(image, axis=axis).copy()
        label = np.flip(label, axis=axis).copy()

        return {'image': image, 'label': label}

class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image = sample['image']
        image = np.expand_dims(image, axis=0).astype(np.float32)
        # image = image.reshape(1, image.shape[0], image.shape[1]).astype(np.float32)
        if 'onehot_label' in sample:
            return {'image': torch.from_numpy(image), 'label': torch.from_numpy(sample['label']).float(),
                    'onehot_label': torch.from_numpy(sample['onehot_label']).float()}
        else:
            return {'image': torch.from_numpy(image), 'label': torch.from_numpy(sample['label']).float()}

def get_data_loader(base_dir: str, split: str, transform: transforms.Compose):
    dataset = base_dir.split('/')[-1]
    assert dataset in ["Prostate_ISBI2013", "Pancreas_MRI", "BraTS2020", "Promise12"], f"dataset: {dataset} is not defined"
    if dataset == "Prostate_ISBI2013":
        return Prostate_ISBI2013(base_dir=base_dir, split=split, transform=transform)
    elif dataset == "Pancreas_MRI":
        return Pancreas_MRI(base_dir=base_dir, split=split, transform=transform)
    elif dataset == "BraTS2020":
        return BraTS2020(base_dir=base_dir, split=split, transform=transform)
    elif dataset == "Promise12":
        return Promise12(base_dir=base_dir, split=split, transform=transform)
