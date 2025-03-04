import os
import numpy as np
from torch.utils.data import Dataset
import SimpleITK as sitk
import torchvision.transforms as transforms
import torch
import random
from torch.nn import functional as F
import sys
sys.path.append("/root/autodl-tmp/SAM_adaptive_learning")
sys.path.append("/root/autodl-tmp/SAM_adaptive_learning/dataloaders")

from tqdm import tqdm
from sklearn.cluster import KMeans
from sampling import *




class Promise12(Dataset):
    """ 
        base_dir: the location of data.
        split: the part used for training, validation or testing.
        transform: augmentation to images.
    """
    def __init__(self, base_dir="/root/autodl-tmp/SAM_adaptive_learning/sam_data/Promise12", split="train", select_ratio=1.0, select_strategy="random", transform=None):
        self.base_dir = base_dir
        self.pixel_mean = torch.Tensor([123.675, 116.28, 103.53]).view(-1, 1, 1)
        self.pixel_std = torch.Tensor([58.395, 57.12, 57.375]).view(-1, 1, 1)
        self.sample_list = []
        self.split = split
        self.transform = transform
        self.sample_list = os.listdir(os.path.join(self.base_dir, "images", split))
        if split == "train":
            if select_strategy == "random":
                self.final_sample_list = random.sample(self.sample_list, int(len(self.sample_list) * select_ratio))
            elif select_strategy == "FPS":
                feats, name_list, num_samples = self.get_parameters(select_ratio)
                self.final_sample_list = select_valuable_samples_FPS(feats, name_list, num_samples)
            elif select_strategy == "TypiClust":
                feats, name_list, num_samples = self.get_parameters(select_ratio)
                self.final_sample_list = select_valuable_samples_TypiClust(feats, name_list, num_samples)
            elif select_strategy == "CALR":
                feats, name_list, num_samples = self.get_parameters(select_ratio)
                self.final_sample_list = select_valuable_samples_CALR(feats, name_list, num_samples)
            elif select_strategy == "ALPS":
                feats, name_list, num_samples = self.get_parameters(select_ratio)
                self.final_sample_list = select_valuable_samples_ALPS(feats, name_list, num_samples)
            elif select_strategy == "ProbCover":
                feats, name_list, num_samples = self.get_parameters(select_ratio)
                self.final_sample_list = select_valuable_samples_ProbCover(feats, name_list, num_samples)
            elif select_strategy == "my_min":
                self.final_sample_list = self._get_min_selected_samples(select_ratio)
            elif select_strategy == "my_max":
                self.final_sample_list = self._get_max_selected_samples(select_ratio)
            elif select_strategy == "my_middle":
                self.final_sample_list = self._get_middle_selected_samples(select_ratio)
            elif select_strategy == "my_greedy":
                self.final_sample_list = self._get_greedy_selected_samples(select_ratio)
            elif select_strategy == "my_cluster_only":
                self.final_sample_list = self._get_cluster_only_selected_samples(select_ratio)
            elif select_strategy == "my_uncertainty_only":
                self.final_sample_list = self._get_uncertainy_only_selected_samples(select_ratio)
            elif select_strategy == "my_half_cluster":
                self.final_sample_list = self._get_half_cluster_with_uncertainty_selected_samples(select_ratio)
            elif select_strategy == "my_half_cluster_far_uncertainty":
                self.final_sample_list = self._get_half_cluster_with_farest_uncertainty_selected_samples(select_ratio)
            elif select_strategy == "my_half_cluster_suitable_uncertainty":
                self.final_sample_list = self._get_half_cluster_with_suitable_uncertainty_selected_samples(select_ratio)
            else:
                self.final_sample_list = self.sample_list
            # Save the selected samples after selection
            self.save_samples(select_ratio, select_strategy)
        else:
            self.final_sample_list = self.sample_list
        
        
    def __len__(self):
        return len(self.final_sample_list)
    

    def __getitem__(self, idx):
        filename = self.final_sample_list[idx]
        file_path = os.path.join(self.base_dir, "images", self.split, filename)
        image = sitk.GetArrayFromImage(sitk.ReadImage(file_path)).astype(np.float32).squeeze()
        image = (image - image.mean()) / image.std()
        label = sitk.GetArrayFromImage(sitk.ReadImage(file_path.replace("images", "labels"))).squeeze().astype(np.uint8)
        label[label > 0] = 1
        sample = {'image': image, 'label': label}
        if self.transform is not None:
            sample = self.transform(sample)
        sample['idx'] = idx
        sample['filename'] = file_path
        
        return sample
    
    def preprocess(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize pixel values and pad to a square input."""
        # Normalize colors
        x = (x - self.pixel_mean) / self.pixel_std

        # Pad
        h, w = x.shape[-2:]
        padh = 1024 - h
        padw = 1024 - w
        x = F.pad(x, (0, padw, 0, padh))
        return x
    
    def get_selected_samples(self, select_ratio):
        from SAM_FineTuning.segment_anything_finetune.modeling.image_encoder import ImageEncoderViT
        image_encoder = ImageEncoderViT()
        state_dict = torch.load("/root/autodl-tmp/SAM_adaptive_learning/SAM_FineTuning/checkpoints/sam_vit_b_01ec64.pth")
        sam_encoder_dict = image_encoder.state_dict()
        state_dict = {k: v for k, v in state_dict.items() if k in sam_encoder_dict}
        sam_encoder_dict.update(state_dict)
        image_encoder.load_state_dict(sam_encoder_dict)
        image_encoder.eval()
        
        results = []
        for filename in tqdm(self.sample_list):
            file_path = os.path.join(self.base_dir, "images", self.split, filename)
            image = sitk.GetArrayFromImage(sitk.ReadImage(file_path)).astype(np.float32).squeeze()
            image = (image - image.mean()) / image.std()
            image = torch.from_numpy(image).unsqueeze(0).unsqueeze(0).repeat(1, 3, 1, 1)  # Add batch and channel dimensions
            image = self.preprocess(image)
            with torch.no_grad():
                image_feature = image_encoder(image).squeeze().numpy()
            
            
            uncertainty_path = file_path.replace("images", "uncertainty").replace(".gz", "_uncertainty.npy")
            uncertainty_score = np.load(uncertainty_path)
            uncertainty_mean = np.mean(uncertainty_score)
            
            global_feature = np.sum(image_feature * uncertainty_score, axis=(1, 2)) / np.sum(uncertainty_score)
            # 保存global_feature
            feature_dir = "/root/autodl-tmp/SAM_adaptive_learning/sam_data/Promise12/feature"
            os.makedirs(feature_dir, exist_ok=True)
            feature_path = os.path.join(feature_dir, filename.replace(".gz", ".npy"))
            np.save(feature_path, global_feature)
            
            results.append([file_path, global_feature, uncertainty_mean])
        
        # 计算挑选个数
        select_num = int(select_ratio * len(results))
        # 按uncertainty_mean排序
        results.sort(key=lambda x: x[2])
        
        # 提取global_feature
        features = np.array([result[1] for result in results])
        
        # k-means聚类
        kmeans = KMeans(n_clusters=select_num, random_state=0).fit(features)
        labels = kmeans.labels_
        
        # 按类别和uncertainty_mean均匀挑选样本
        selected_samples = []
        for i in range(select_num):
            cluster_indices = np.where(labels == i)[0]
            cluster_samples = [results[idx] for idx in cluster_indices]
            cluster_samples.sort(key=lambda x: x[2])
            selected_samples.append(os.path.basename(cluster_samples[len(cluster_samples) // 2][0]))  # 选择中间的样本的filename
        
        # 将挑选到的样本路径保存在txt文件中
        with open(os.path.join("/root/autodl-fs/SAM_adaptive_learning/model/Promise12/train/0.1/my/2026/sample.txt"), "w") as f:
            for sample in selected_samples:
                f.write(sample + "\n")
        
        return selected_samples

    def _get_middle_selected_samples(self, select_ratio): # 中心挑选
        results = []
        for filename in tqdm(self.sample_list):
            file_path = os.path.join(self.base_dir, "images", self.split, filename)
            
            
            uncertainty_path = file_path.replace("images", "uncertainty").replace(".gz", "_uncertainty.npy")
            feature_path = file_path.replace("images/train", "feature").replace(".gz", ".npy")
            uncertainty_score = np.load(uncertainty_path)
            uncertainty_mean = np.mean(uncertainty_score)
            global_feature = np.load(feature_path)
            
            results.append([file_path, global_feature, uncertainty_mean])
        
        # 计算挑选个数
        select_num = int(select_ratio * len(results))
        # 按uncertainty_mean排序
        results.sort(key=lambda x: x[2])
        
        # 提取global_feature
        features = np.array([result[1] for result in results])
        
        # k-means聚类
        kmeans = KMeans(n_clusters=select_num, random_state=0).fit(features)
        labels = kmeans.labels_
        
        # 按类别和uncertainty_mean均匀挑选样本
        selected_samples = []
        for i in range(select_num):
            cluster_indices = np.where(labels == i)[0]
            cluster_samples = [results[idx] for idx in cluster_indices]
            cluster_samples.sort(key=lambda x: x[2])
            selected_samples.append(os.path.basename(cluster_samples[len(cluster_samples) // 2][0]))  # 选择中间的样本的filename
        
       
        
        return selected_samples

    def _get_greedy_selected_samples(self, select_ratio): # 贪心
        results = []
        for filename in tqdm(self.sample_list):
            file_path = os.path.join(self.base_dir, "images", self.split, filename)
            
            uncertainty_path = file_path.replace("images", "uncertainty").replace(".gz", "_uncertainty.npy")
            feature_path = file_path.replace("images/train", "feature").replace(".gz", ".npy")
            uncertainty_score = np.load(uncertainty_path)
            uncertainty_mean = np.mean(uncertainty_score)
            global_feature = np.load(feature_path)
            
            results.append([file_path, global_feature, uncertainty_mean])
        
        # 计算挑选个数
        select_num = int(select_ratio * len(results))
        # 按uncertainty_mean排序
        results.sort(key=lambda x: x[2])
        
        # 提取global_feature
        features = np.array([result[1] for result in results])
        
        # k-means聚类
        kmeans = KMeans(n_clusters=select_num, random_state=0).fit(features)
        labels = kmeans.labels_

        # 贪心选择样本
        selected_samples = []
        selected_features = []
        
        # 从每个cluster中选择一个样本
        for i in range(select_num):
            cluster_indices = np.where(labels == i)[0]
            cluster_samples = [results[idx] for idx in cluster_indices]
            cluster_features = np.array([sample[1] for sample in cluster_samples])
            
            if len(selected_features) == 0:
                # 如果是第一个样本，选择uncertainty最中等的
                mid_idx = len(cluster_samples) // 2
                selected_samples.append(os.path.basename(cluster_samples[mid_idx][0]))
                selected_features.append(cluster_features[mid_idx])
            else:
                # 计算每个候选样本与已选样本的最小距离
                selected_features_array = np.array(selected_features)
                min_distances = []
                for feat in cluster_features:
                    distances = np.linalg.norm(selected_features_array - feat, axis=1)
                    min_distances.append(np.min(distances))
                
                # 选择与已有样本距离最远的样本
                max_dist_idx = np.argmax(min_distances)
                selected_samples.append(os.path.basename(cluster_samples[max_dist_idx][0]))
                selected_features.append(cluster_features[max_dist_idx])
        
        
        return selected_samples

    def _get_max_selected_samples(self, select_ratio): # 高uncertainty
        results = []
        for filename in tqdm(self.sample_list):
            file_path = os.path.join(self.base_dir, "images", self.split, filename)
            
            uncertainty_path = file_path.replace("images", "uncertainty").replace(".gz", "_uncertainty.npy")
            feature_path = file_path.replace("images/train", "feature").replace(".gz", ".npy")
            uncertainty_score = np.load(uncertainty_path)
            uncertainty_mean = np.mean(uncertainty_score)
            global_feature = np.load(feature_path)
            
            results.append([file_path, global_feature, uncertainty_mean])
        
        # 计算挑选个数
        select_num = int(select_ratio * len(results))
        # 按uncertainty_mean排序
        results.sort(key=lambda x: x[2])
        
        # 提取global_feature
        features = np.array([result[1] for result in results])
        
        # k-means聚类
        kmeans = KMeans(n_clusters=select_num, random_state=0).fit(features)
        labels = kmeans.labels_

        # 从每个cluster中选择一个样本
        selected_samples = []
        
        for i in range(select_num):
            cluster_indices = np.where(labels == i)[0]
            # 选择index最高的样本
            highest_idx = np.max(cluster_indices)
            selected_samples.append(os.path.basename(results[highest_idx][0]))

        return selected_samples

    def _get_min_selected_samples(self, select_ratio): # 低uncertainty
        results = []
        for filename in tqdm(self.sample_list):
            file_path = os.path.join(self.base_dir, "images", self.split, filename)
            
            uncertainty_path = file_path.replace("images", "uncertainty").replace(".gz", "_uncertainty.npy")
            feature_path = file_path.replace("images/train", "feature").replace(".gz", ".npy")
            uncertainty_score = np.load(uncertainty_path)
            uncertainty_mean = np.mean(uncertainty_score)
            global_feature = np.load(feature_path)
            
            results.append([file_path, global_feature, uncertainty_mean])
        
        # 计算挑选个数
        select_num = int(select_ratio * len(results))
        # 按uncertainty_mean排序
        results.sort(key=lambda x: x[2])
        
        # 提取global_feature
        features = np.array([result[1] for result in results])
        
        # k-means聚类
        kmeans = KMeans(n_clusters=select_num, random_state=0).fit(features)
        labels = kmeans.labels_

        # 从每个cluster中选择一个样本
        selected_samples = []
        
        for i in range(select_num):
            cluster_indices = np.where(labels == i)[0]
            # 选择index最高的样本
            highest_idx = np.max(cluster_indices)
            selected_samples.append(os.path.basename(results[highest_idx][0]))

        return selected_samples

    def _get_cluster_only_selected_samples(self, select_ratio): # cluster only
        results = []
        for filename in tqdm(self.sample_list):
            file_path = os.path.join(self.base_dir, "images", self.split, filename)
            
            
            uncertainty_path = file_path.replace("images", "uncertainty").replace(".gz", "_uncertainty.npy")
            feature_path = file_path.replace("images/train", "feature").replace(".gz", ".npy")
            uncertainty_score = np.load(uncertainty_path)
            uncertainty_mean = np.mean(uncertainty_score)
            global_feature = np.load(feature_path)
            
            results.append([file_path, global_feature, uncertainty_mean])
        
        # 计算挑选个数
        select_num = int(select_ratio * len(results))
        # 按uncertainty_mean排序
        results.sort(key=lambda x: x[2])
        
        # 提取global_feature
        features = np.array([result[1] for result in results])
        
        # k-means聚类
        kmeans = KMeans(n_clusters=select_num, random_state=0).fit(features)
        labels = kmeans.labels_
        
        # 按类别和uncertainty_mean均匀挑选样本
        selected_samples = []
        for i in range(select_num):
            cluster_indices = np.where(labels == i)[0]
            cluster_samples = [results[idx] for idx in cluster_indices]
            cluster_samples.sort(key=lambda x: x[2])
            selected_sample = random.choice(cluster_samples)
            selected_samples.append(os.path.basename(selected_sample[0]))
       
        
        return selected_samples

    def _get_uncertainy_only_selected_samples(self, select_ratio): # uncertainty only
        results = []
        for filename in tqdm(self.sample_list):
            file_path = os.path.join(self.base_dir, "images", self.split, filename)
            
            uncertainty_path = file_path.replace("images", "uncertainty").replace(".gz", "_uncertainty.npy")
            feature_path = file_path.replace("images/train", "feature").replace(".gz", ".npy")
            uncertainty_score = np.load(uncertainty_path)
            uncertainty_mean = np.mean(uncertainty_score)
            global_feature = np.load(feature_path)
            
            results.append([file_path, global_feature, uncertainty_mean])
        
        # 计算挑选个数
        select_num = int(select_ratio * len(results))
        
        # 按uncertainty_mean排序
        results.sort(key=lambda x: x[2])
        
        # 均匀选择样本
        step = len(results) // select_num
        selected_samples = []
        for i in range(select_num):
            idx = i * step
            selected_samples.append(os.path.basename(results[idx][0]))
        
        return selected_samples

    def _get_half_cluster_with_uncertainty_selected_samples(self, select_ratio): # half cluster with uncertainty
        results = []
        for filename in tqdm(self.sample_list):
            file_path = os.path.join(self.base_dir, "images", self.split, filename)
            
            uncertainty_path = file_path.replace("images", "uncertainty").replace(".gz", "_uncertainty.npy")
            feature_path = file_path.replace("images/train", "feature").replace(".gz", ".npy")
            uncertainty_score = np.load(uncertainty_path)
            uncertainty_mean = np.mean(uncertainty_score)
            global_feature = np.load(feature_path)
            
            results.append([file_path, global_feature, uncertainty_mean])
        
        # 计算挑选个数
        select_num = int(select_ratio * len(results))
        # 设置聚类数为select_num的一半
        n_clusters = select_num // 2
        
        # 提取global_feature
        features = np.array([result[1] for result in results])
        
        # k-means聚类
        kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(features)
        labels = kmeans.labels_
        
        # 从每个cluster中选择样本
        selected_samples = []
        remaining_samples = select_num
        
        # 计算每个簇应选择的样本数
        samples_per_cluster = remaining_samples // n_clusters
        extra_samples = remaining_samples % n_clusters
        
        # 按uncertainty_mean排序
        results.sort(key=lambda x: x[2])
        
        # 从每个簇中选择样本
        for i in range(n_clusters):
            cluster_indices = np.where(labels == i)[0]
            cluster_samples = [results[idx] for idx in cluster_indices]
            
            # 按uncertainty_mean排序
            cluster_samples.sort(key=lambda x: x[2])
            
            # 确定当前簇需要选择的样本数
            n_select = samples_per_cluster + (1 if i < extra_samples else 0)
            
            # 均匀选择样本
            step = len(cluster_samples) // n_select
            for j in range(n_select):
                idx = j * step
                if idx >= len(cluster_samples):
                    idx = len(cluster_samples) - 1
                selected_samples.append(os.path.basename(cluster_samples[idx][0]))
    
        return selected_samples

    def _get_half_cluster_with_farest_uncertainty_selected_samples(self, select_ratio): # half cluster with uncertainty
        results = []
        for filename in tqdm(self.sample_list):
            file_path = os.path.join(self.base_dir, "images", self.split, filename)
            
            uncertainty_path = file_path.replace("images", "uncertainty").replace(".gz", "_uncertainty.npy")
            feature_path = file_path.replace("images/train", "feature").replace(".gz", ".npy")
            uncertainty_score = np.load(uncertainty_path)
            uncertainty_mean = np.mean(uncertainty_score)
            global_feature = np.load(feature_path)
            
            results.append([file_path, global_feature, uncertainty_mean])
        
        # 计算挑选个数
        select_num = int(select_ratio * len(results))
        # 设置聚类数为select_num的一半
        n_clusters = select_num // 2
        
        # 提取global_feature
        features = np.array([result[1] for result in results])
        
        # k-means聚类
        kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(features)
        labels = kmeans.labels_
        
        # 从每个cluster中选择样本
        selected_samples = []
        remaining_samples = select_num
        
        # 计算每个簇应选择的样本数
        samples_per_cluster = remaining_samples // n_clusters
        extra_samples = remaining_samples % n_clusters
    
        
        # 从每个簇中选择样本
        for i in range(n_clusters):
            cluster_indices = np.where(labels == i)[0]
            cluster_samples = [results[idx] for idx in cluster_indices]
            
            # 按uncertainty_mean排序
            cluster_samples.sort(key=lambda x: x[2])
            
            # 确定当前簇需要选择的样本数
            n_select = samples_per_cluster + (1 if i < extra_samples else 0)
            
            if n_select == 3:
                # 如果需要选择三个样本，选择uncertainty最中等的
                mid_idx = len(cluster_samples) // 2
                selected_samples.append(os.path.basename(cluster_samples[mid_idx][0]))
                selected_samples.append(os.path.basename(cluster_samples[0][0]))  # 最小的uncertainty
                selected_samples.append(os.path.basename(cluster_samples[-1][0]))  # 最大的uncertainty
            else:
                # 如果需要选择两个样本，选择uncertainty最大和最小的
                selected_samples.append(os.path.basename(cluster_samples[0][0]))  # 最小的uncertainty
                selected_samples.append(os.path.basename(cluster_samples[-1][0]))  # 最大的uncertainty
        
        return selected_samples

    def _get_half_cluster_with_suitable_uncertainty_selected_samples(self, select_ratio): # half cluster with uncertainty
        results = []
        for filename in tqdm(self.sample_list):
            file_path = os.path.join(self.base_dir, "images", self.split, filename)
            
            uncertainty_path = file_path.replace("images", "uncertainty").replace(".gz", "_uncertainty.npy")
            feature_path = file_path.replace("images/train", "feature").replace(".gz", ".npy")
            uncertainty_score = np.load(uncertainty_path)
            uncertainty_mean = np.mean(uncertainty_score)
            global_feature = np.load(feature_path)
            
            results.append([file_path, global_feature, uncertainty_mean])
        
        # 计算挑选个数
        select_num = int(select_ratio * len(results))
        # 设置聚类数为select_num的一半
        n_clusters = select_num // 2
        
        # 提取global_feature
        features = np.array([result[1] for result in results])
        
        # k-means聚类
        kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(features)
        labels = kmeans.labels_
        
        # 从每个cluster中选择样本
        selected_samples = []
        remaining_samples = select_num
        
        # 计算每个簇应选择的样本数
        samples_per_cluster = remaining_samples // n_clusters
        extra_samples = remaining_samples % n_clusters
    
        
        # 从每个簇中选择样本
        for i in range(n_clusters):
            cluster_indices = np.where(labels == i)[0]
            cluster_samples = [results[idx] for idx in cluster_indices]
            
            # 按uncertainty_mean排序
            cluster_samples.sort(key=lambda x: x[2])
            
            # 确定当前簇需要选择的样本数
            n_select = samples_per_cluster + (1 if i < extra_samples else 0)
            
            if n_select == 3:
                # 如果需要选择三个样本，选择uncertainty最中等的
                idx1, idx2, idx3 = len(cluster_samples) // 4, len(cluster_samples) // 2, len(cluster_samples) * 3 // 4
                selected_samples.append(os.path.basename(cluster_samples[idx1][0]))
                selected_samples.append(os.path.basename(cluster_samples[idx2][0]))
                selected_samples.append(os.path.basename(cluster_samples[idx3][0])) 
            else:
                # 如果需要选择两个样本，选择uncertainty最大和最小的
                idx1, idx2 = len(cluster_samples) // 3, len(cluster_samples) * 2 // 3
                selected_samples.append(os.path.basename(cluster_samples[idx1][0]))  # 最小的uncertainty
                selected_samples.append(os.path.basename(cluster_samples[idx2][0]))  # 最大的uncertainty
        
        return selected_samples



    def get_parameters(self, select_ratio):
        """Get parameters needed for sample selection strategies."""
        results = []
        for filename in self.sample_list:
            file_path = os.path.join(self.base_dir, "images", self.split, filename)
            
            feature_path = file_path.replace("images/train", "feature").replace(".gz", ".npy")
            global_feature = np.load(feature_path)
            
            results.append([filename, global_feature])
            
        feats = np.array([result[1] for result in results])
        name_list = np.array([result[0] for result in results])
        num_samples = int(select_ratio * len(results))
        
        return feats, name_list, num_samples

    def save_samples(self, select_ratio, select_strategy):
        """Save selected samples to a text file."""
        save_dir = f"/root/autodl-fs/SAM_adaptive_learning/model/Promise12/train/{select_ratio}/{select_strategy}/2026"
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, "sample.txt")
        with open(save_path, "w") as f:
            for sample in self.final_sample_list:
                f.write(f"{sample}\n")
    

class UTAH(Dataset):
    """ 
        base_dir: the location of data.
        split: the part used for training, validation or testing.
        transform: augmentation to images.
    """
    def __init__(self, base_dir="/root/autodl-tmp/SAM_adaptive_learning/sam_data/UTAH", split="train", select_ratio=1.0, select_strategy="random", transform=None):
        self.base_dir = base_dir
        self.pixel_mean = torch.Tensor([123.675, 116.28, 103.53]).view(-1, 1, 1)
        self.pixel_std = torch.Tensor([58.395, 57.12, 57.375]).view(-1, 1, 1)
        self.sample_list = []
        self.split = split
        self.transform = transform
        self.sample_list = os.listdir(os.path.join(self.base_dir, "images", split))
        if split == "train":
            if select_strategy == "random":
                self.final_sample_list = random.sample(self.sample_list, int(len(self.sample_list) * select_ratio))
            elif select_strategy == "FPS":
                feats, name_list, num_samples = self.get_parameters(select_ratio)
                self.final_sample_list = select_valuable_samples_FPS(feats, name_list, num_samples)
            elif select_strategy == "TypiClust":
                feats, name_list, num_samples = self.get_parameters(select_ratio)
                self.final_sample_list = select_valuable_samples_TypiClust(feats, name_list, num_samples)
            elif select_strategy == "CALR":
                feats, name_list, num_samples = self.get_parameters(select_ratio)
                self.final_sample_list = select_valuable_samples_CALR(feats, name_list, num_samples)
            elif select_strategy == "ALPS":
                feats, name_list, num_samples = self.get_parameters(select_ratio)
                self.final_sample_list = select_valuable_samples_ALPS(feats, name_list, num_samples)
            elif select_strategy == "ProbCover":
                feats, name_list, num_samples = self.get_parameters(select_ratio)
                self.final_sample_list = select_valuable_samples_ProbCover(feats, name_list, num_samples)
            elif select_strategy == "my_min":
                self.final_sample_list = self._get_min_selected_samples(select_ratio)
            elif select_strategy == "my_max":
                self.final_sample_list = self._get_max_selected_samples(select_ratio)
            elif select_strategy == "my_middle":
                self.final_sample_list = self._get_middle_selected_samples(select_ratio)
            elif select_strategy == "my_greedy":
                self.final_sample_list = self._get_greedy_selected_samples(select_ratio)
            elif select_strategy == "my_cluster_only":
                self.final_sample_list = self._get_cluster_only_selected_samples(select_ratio)
            elif select_strategy == "my_uncertainty_only":
                self.final_sample_list = self._get_uncertainy_only_selected_samples(select_ratio)
            elif select_strategy == "my_half_cluster":
                self.final_sample_list = self._get_half_cluster_with_uncertainty_selected_samples(select_ratio)
            elif select_strategy == "my_half_cluster_far_uncertainty":
                self.final_sample_list = self._get_half_cluster_with_farest_uncertainty_selected_samples(select_ratio)
            elif select_strategy == "my_half_cluster_suitable_uncertainty":
                self.final_sample_list = self._get_half_cluster_with_suitable_uncertainty_selected_samples(select_ratio)
            else:
                self.final_sample_list = self.sample_list
            # Save the selected samples after selection
            self.save_samples(select_ratio, select_strategy)
        else:
            self.final_sample_list = self.sample_list
        
        
    def __len__(self):
        return len(self.final_sample_list)
    

    def __getitem__(self, idx):
        filename = self.final_sample_list[idx]
        file_path = os.path.join(self.base_dir, "images", self.split, filename)
        image = sitk.GetArrayFromImage(sitk.ReadImage(file_path)).astype(np.float32).squeeze()
        image = (image - image.mean()) / image.std()
        label = sitk.GetArrayFromImage(sitk.ReadImage(file_path.replace("images", "labels"))).squeeze().astype(np.uint8)
        label[label > 0] = 1
        sample = {'image': image, 'label': label}
        if self.transform is not None:
            sample = self.transform(sample)
        sample['idx'] = idx
        sample['filename'] = file_path
        
        return sample
    
    def preprocess(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize pixel values and pad to a square input."""
        # Normalize colors
        x = (x - self.pixel_mean) / self.pixel_std

        # Pad
        h, w = x.shape[-2:]
        padh = 1024 - h
        padw = 1024 - w
        x = F.pad(x, (0, padw, 0, padh))
        return x
    
    def get_selected_samples(self, select_ratio):
        from SAM_FineTuning.segment_anything_finetune.modeling.image_encoder import ImageEncoderViT
        image_encoder = ImageEncoderViT()
        state_dict = torch.load("/root/autodl-tmp/SAM_adaptive_learning/SAM_FineTuning/checkpoints/sam_vit_b_01ec64.pth")
        sam_encoder_dict = image_encoder.state_dict()
        state_dict = {k: v for k, v in state_dict.items() if k in sam_encoder_dict}
        sam_encoder_dict.update(state_dict)
        image_encoder.load_state_dict(sam_encoder_dict)
        image_encoder.eval()
        
        results = []
        for filename in tqdm(self.sample_list):
            file_path = os.path.join(self.base_dir, "images", self.split, filename)
            image = sitk.GetArrayFromImage(sitk.ReadImage(file_path)).astype(np.float32).squeeze()
            image = (image - image.mean()) / image.std()
            image = torch.from_numpy(image).unsqueeze(0).unsqueeze(0).repeat(1, 3, 1, 1)  # Add batch and channel dimensions
            image = self.preprocess(image)
            with torch.no_grad():
                image_feature = image_encoder(image).squeeze().numpy()
            
            
            uncertainty_path = file_path.replace("images", "uncertainty").replace(".gz", "_uncertainty.npy")
            uncertainty_score = np.load(uncertainty_path)
            uncertainty_mean = np.mean(uncertainty_score)
            
            global_feature = np.sum(image_feature * uncertainty_score, axis=(1, 2)) / np.sum(uncertainty_score)
            # 保存global_feature
            feature_dir = "/root/autodl-tmp/SAM_adaptive_learning/sam_data/UTAH/feature"
            os.makedirs(feature_dir, exist_ok=True)
            feature_path = os.path.join(feature_dir, filename.replace(".gz", ".npy"))
            np.save(feature_path, global_feature)
            
            results.append([file_path, global_feature, uncertainty_mean])
        
        # 计算挑选个数
        select_num = int(select_ratio * len(results))
        # 按uncertainty_mean排序
        results.sort(key=lambda x: x[2])
        
        # 提取global_feature
        features = np.array([result[1] for result in results])
        
        # k-means聚类
        kmeans = KMeans(n_clusters=select_num, random_state=0).fit(features)
        labels = kmeans.labels_
        
        # 按类别和uncertainty_mean均匀挑选样本
        selected_samples = []
        for i in range(select_num):
            cluster_indices = np.where(labels == i)[0]
            cluster_samples = [results[idx] for idx in cluster_indices]
            cluster_samples.sort(key=lambda x: x[2])
            selected_samples.append(os.path.basename(cluster_samples[len(cluster_samples) // 2][0]))  # 选择中间的样本的filename
        
        # 将挑选到的样本路径保存在txt文件中
        with open(os.path.join("/root/autodl-fs/SAM_adaptive_learning/model/UTAH/train/0.1/my/2026/sample.txt"), "w") as f:
            for sample in selected_samples:
                f.write(sample + "\n")
        
        return selected_samples

    def _get_middle_selected_samples(self, select_ratio): # 中心挑选
        results = []
        for filename in tqdm(self.sample_list):
            file_path = os.path.join(self.base_dir, "images", self.split, filename)
            
            
            uncertainty_path = file_path.replace("images", "uncertainty").replace(".gz", "_uncertainty.npy")
            feature_path = file_path.replace("images/train", "feature").replace(".gz", ".npy")
            uncertainty_score = np.load(uncertainty_path)
            uncertainty_mean = np.mean(uncertainty_score)
            global_feature = np.load(feature_path)
            
            results.append([file_path, global_feature, uncertainty_mean])
        
        # 计算挑选个数
        select_num = int(select_ratio * len(results))
        # 按uncertainty_mean排序
        results.sort(key=lambda x: x[2])
        
        # 提取global_feature
        features = np.array([result[1] for result in results])
        
        # k-means聚类
        kmeans = KMeans(n_clusters=select_num, random_state=0).fit(features)
        labels = kmeans.labels_
        
        # 按类别和uncertainty_mean均匀挑选样本
        selected_samples = []
        for i in range(select_num):
            cluster_indices = np.where(labels == i)[0]
            cluster_samples = [results[idx] for idx in cluster_indices]
            cluster_samples.sort(key=lambda x: x[2])
            selected_samples.append(os.path.basename(cluster_samples[len(cluster_samples) // 2][0]))  # 选择中间的样本的filename
        
       
        
        return selected_samples

    def _get_greedy_selected_samples(self, select_ratio): # 贪心
        results = []
        for filename in tqdm(self.sample_list):
            file_path = os.path.join(self.base_dir, "images", self.split, filename)
            
            uncertainty_path = file_path.replace("images", "uncertainty").replace(".gz", "_uncertainty.npy")
            feature_path = file_path.replace("images/train", "feature").replace(".gz", ".npy")
            uncertainty_score = np.load(uncertainty_path)
            uncertainty_mean = np.mean(uncertainty_score)
            global_feature = np.load(feature_path)
            
            results.append([file_path, global_feature, uncertainty_mean])
        
        # 计算挑选个数
        select_num = int(select_ratio * len(results))
        # 按uncertainty_mean排序
        results.sort(key=lambda x: x[2])
        
        # 提取global_feature
        features = np.array([result[1] for result in results])
        
        # k-means聚类
        kmeans = KMeans(n_clusters=select_num, random_state=0).fit(features)
        labels = kmeans.labels_

        # 贪心选择样本
        selected_samples = []
        selected_features = []
        
        # 从每个cluster中选择一个样本
        for i in range(select_num):
            cluster_indices = np.where(labels == i)[0]
            cluster_samples = [results[idx] for idx in cluster_indices]
            cluster_features = np.array([sample[1] for sample in cluster_samples])
            
            if len(selected_features) == 0:
                # 如果是第一个样本，选择uncertainty最中等的
                mid_idx = len(cluster_samples) // 2
                selected_samples.append(os.path.basename(cluster_samples[mid_idx][0]))
                selected_features.append(cluster_features[mid_idx])
            else:
                # 计算每个候选样本与已选样本的最小距离
                selected_features_array = np.array(selected_features)
                min_distances = []
                for feat in cluster_features:
                    distances = np.linalg.norm(selected_features_array - feat, axis=1)
                    min_distances.append(np.min(distances))
                
                # 选择与已有样本距离最远的样本
                max_dist_idx = np.argmax(min_distances)
                selected_samples.append(os.path.basename(cluster_samples[max_dist_idx][0]))
                selected_features.append(cluster_features[max_dist_idx])
        
        
        return selected_samples

    def _get_max_selected_samples(self, select_ratio): # 高uncertainty
        results = []
        for filename in tqdm(self.sample_list):
            file_path = os.path.join(self.base_dir, "images", self.split, filename)
            
            uncertainty_path = file_path.replace("images", "uncertainty").replace(".gz", "_uncertainty.npy")
            feature_path = file_path.replace("images/train", "feature").replace(".gz", ".npy")
            uncertainty_score = np.load(uncertainty_path)
            uncertainty_mean = np.mean(uncertainty_score)
            global_feature = np.load(feature_path)
            
            results.append([file_path, global_feature, uncertainty_mean])
        
        # 计算挑选个数
        select_num = int(select_ratio * len(results))
        # 按uncertainty_mean排序
        results.sort(key=lambda x: x[2])
        
        # 提取global_feature
        features = np.array([result[1] for result in results])
        
        # k-means聚类
        kmeans = KMeans(n_clusters=select_num, random_state=0).fit(features)
        labels = kmeans.labels_

        # 从每个cluster中选择一个样本
        selected_samples = []
        
        for i in range(select_num):
            cluster_indices = np.where(labels == i)[0]
            # 选择index最高的样本
            highest_idx = np.max(cluster_indices)
            selected_samples.append(os.path.basename(results[highest_idx][0]))

        return selected_samples

    def _get_min_selected_samples(self, select_ratio): # 低uncertainty
        results = []
        for filename in tqdm(self.sample_list):
            file_path = os.path.join(self.base_dir, "images", self.split, filename)
            
            uncertainty_path = file_path.replace("images", "uncertainty").replace(".gz", "_uncertainty.npy")
            feature_path = file_path.replace("images/train", "feature").replace(".gz", ".npy")
            uncertainty_score = np.load(uncertainty_path)
            uncertainty_mean = np.mean(uncertainty_score)
            global_feature = np.load(feature_path)
            
            results.append([file_path, global_feature, uncertainty_mean])
        
        # 计算挑选个数
        select_num = int(select_ratio * len(results))
        # 按uncertainty_mean排序
        results.sort(key=lambda x: x[2])
        
        # 提取global_feature
        features = np.array([result[1] for result in results])
        
        # k-means聚类
        kmeans = KMeans(n_clusters=select_num, random_state=0).fit(features)
        labels = kmeans.labels_

        # 从每个cluster中选择一个样本
        selected_samples = []
        
        for i in range(select_num):
            cluster_indices = np.where(labels == i)[0]
            # 选择index最高的样本
            highest_idx = np.max(cluster_indices)
            selected_samples.append(os.path.basename(results[highest_idx][0]))

        return selected_samples

    def _get_cluster_only_selected_samples(self, select_ratio): # cluster only
        results = []
        for filename in tqdm(self.sample_list):
            file_path = os.path.join(self.base_dir, "images", self.split, filename)
            
            
            uncertainty_path = file_path.replace("images", "uncertainty").replace(".gz", "_uncertainty.npy")
            feature_path = file_path.replace("images/train", "feature").replace(".gz", ".npy")
            uncertainty_score = np.load(uncertainty_path)
            uncertainty_mean = np.mean(uncertainty_score)
            global_feature = np.load(feature_path)
            
            results.append([file_path, global_feature, uncertainty_mean])
        
        # 计算挑选个数
        select_num = int(select_ratio * len(results))
        # 按uncertainty_mean排序
        results.sort(key=lambda x: x[2])
        
        # 提取global_feature
        features = np.array([result[1] for result in results])
        
        # k-means聚类
        kmeans = KMeans(n_clusters=select_num, random_state=0).fit(features)
        labels = kmeans.labels_
        
        # 按类别和uncertainty_mean均匀挑选样本
        selected_samples = []
        for i in range(select_num):
            cluster_indices = np.where(labels == i)[0]
            cluster_samples = [results[idx] for idx in cluster_indices]
            cluster_samples.sort(key=lambda x: x[2])
            selected_sample = random.choice(cluster_samples)
            selected_samples.append(os.path.basename(selected_sample[0]))
       
        
        return selected_samples

    def _get_uncertainy_only_selected_samples(self, select_ratio): # uncertainty only
        results = []
        for filename in tqdm(self.sample_list):
            file_path = os.path.join(self.base_dir, "images", self.split, filename)
            
            uncertainty_path = file_path.replace("images", "uncertainty").replace(".gz", "_uncertainty.npy")
            feature_path = file_path.replace("images/train", "feature").replace(".gz", ".npy")
            uncertainty_score = np.load(uncertainty_path)
            uncertainty_mean = np.mean(uncertainty_score)
            global_feature = np.load(feature_path)
            
            results.append([file_path, global_feature, uncertainty_mean])
        
        # 计算挑选个数
        select_num = int(select_ratio * len(results))
        
        # 按uncertainty_mean排序
        results.sort(key=lambda x: x[2])
        
        # 均匀选择样本
        step = len(results) // select_num
        selected_samples = []
        for i in range(select_num):
            idx = i * step
            selected_samples.append(os.path.basename(results[idx][0]))
        
        return selected_samples

    def _get_half_cluster_with_uncertainty_selected_samples(self, select_ratio): # half cluster with uncertainty
        results = []
        for filename in tqdm(self.sample_list):
            file_path = os.path.join(self.base_dir, "images", self.split, filename)
            
            uncertainty_path = file_path.replace("images", "uncertainty").replace(".gz", "_uncertainty.npy")
            feature_path = file_path.replace("images/train", "feature").replace(".gz", ".npy")
            uncertainty_score = np.load(uncertainty_path)
            uncertainty_mean = np.mean(uncertainty_score)
            global_feature = np.load(feature_path)
            
            results.append([file_path, global_feature, uncertainty_mean])
        
        # 计算挑选个数
        select_num = int(select_ratio * len(results))
        # 设置聚类数为select_num的一半
        n_clusters = select_num // 2
        
        # 提取global_feature
        features = np.array([result[1] for result in results])
        
        # k-means聚类
        kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(features)
        labels = kmeans.labels_
        
        # 从每个cluster中选择样本
        selected_samples = []
        remaining_samples = select_num
        
        # 计算每个簇应选择的样本数
        samples_per_cluster = remaining_samples // n_clusters
        extra_samples = remaining_samples % n_clusters
        
        # 按uncertainty_mean排序
        results.sort(key=lambda x: x[2])
        
        # 从每个簇中选择样本
        for i in range(n_clusters):
            cluster_indices = np.where(labels == i)[0]
            cluster_samples = [results[idx] for idx in cluster_indices]
            
            # 按uncertainty_mean排序
            cluster_samples.sort(key=lambda x: x[2])
            
            # 确定当前簇需要选择的样本数
            n_select = samples_per_cluster + (1 if i < extra_samples else 0)
            
            # 均匀选择样本
            step = len(cluster_samples) // n_select
            for j in range(n_select):
                idx = j * step
                if idx >= len(cluster_samples):
                    idx = len(cluster_samples) - 1
                selected_samples.append(os.path.basename(cluster_samples[idx][0]))
    
        return selected_samples

    def _get_half_cluster_with_farest_uncertainty_selected_samples(self, select_ratio): # half cluster with uncertainty
        results = []
        for filename in tqdm(self.sample_list):
            file_path = os.path.join(self.base_dir, "images", self.split, filename)
            
            uncertainty_path = file_path.replace("images", "uncertainty").replace(".gz", "_uncertainty.npy")
            feature_path = file_path.replace("images/train", "feature").replace(".gz", ".npy")
            uncertainty_score = np.load(uncertainty_path)
            uncertainty_mean = np.mean(uncertainty_score)
            global_feature = np.load(feature_path)
            
            results.append([file_path, global_feature, uncertainty_mean])
        
        # 计算挑选个数
        select_num = int(select_ratio * len(results))
        # 设置聚类数为select_num的一半
        n_clusters = select_num // 2
        
        # 提取global_feature
        features = np.array([result[1] for result in results])
        
        # k-means聚类
        kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(features)
        labels = kmeans.labels_
        
        # 从每个cluster中选择样本
        selected_samples = []
        remaining_samples = select_num
        
        # 计算每个簇应选择的样本数
        samples_per_cluster = remaining_samples // n_clusters
        extra_samples = remaining_samples % n_clusters
    
        
        # 从每个簇中选择样本
        for i in range(n_clusters):
            cluster_indices = np.where(labels == i)[0]
            cluster_samples = [results[idx] for idx in cluster_indices]
            
            # 按uncertainty_mean排序
            cluster_samples.sort(key=lambda x: x[2])
            
            # 确定当前簇需要选择的样本数
            n_select = samples_per_cluster + (1 if i < extra_samples else 0)
            
            if n_select == 3:
                # 如果需要选择三个样本，选择uncertainty最中等的
                mid_idx = len(cluster_samples) // 2
                selected_samples.append(os.path.basename(cluster_samples[mid_idx][0]))
                selected_samples.append(os.path.basename(cluster_samples[0][0]))  # 最小的uncertainty
                selected_samples.append(os.path.basename(cluster_samples[-1][0]))  # 最大的uncertainty
            else:
                # 如果需要选择两个样本，选择uncertainty最大和最小的
                selected_samples.append(os.path.basename(cluster_samples[0][0]))  # 最小的uncertainty
                selected_samples.append(os.path.basename(cluster_samples[-1][0]))  # 最大的uncertainty
        
        return selected_samples

    def _get_half_cluster_with_suitable_uncertainty_selected_samples(self, select_ratio): # half cluster with uncertainty
        results = []
        for filename in tqdm(self.sample_list):
            file_path = os.path.join(self.base_dir, "images", self.split, filename)
            
            uncertainty_path = file_path.replace("images", "uncertainty").replace(".gz", "_uncertainty.npy")
            feature_path = file_path.replace("images/train", "feature").replace(".gz", ".npy")
            uncertainty_score = np.load(uncertainty_path)
            uncertainty_mean = np.mean(uncertainty_score)
            global_feature = np.load(feature_path)
            
            results.append([file_path, global_feature, uncertainty_mean])
        
        # 计算挑选个数
        select_num = int(select_ratio * len(results))
        # 设置聚类数为select_num的一半
        n_clusters = select_num // 2
        
        # 提取global_feature
        features = np.array([result[1] for result in results])
        
        # k-means聚类
        kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(features)
        labels = kmeans.labels_
        
        # 从每个cluster中选择样本
        selected_samples = []
        remaining_samples = select_num
        
        # 计算每个簇应选择的样本数
        samples_per_cluster = remaining_samples // n_clusters
        extra_samples = remaining_samples % n_clusters
    
        
        # 从每个簇中选择样本
        for i in range(n_clusters):
            cluster_indices = np.where(labels == i)[0]
            cluster_samples = [results[idx] for idx in cluster_indices]
            
            # 按uncertainty_mean排序
            cluster_samples.sort(key=lambda x: x[2])
            
            # 确定当前簇需要选择的样本数
            n_select = samples_per_cluster + (1 if i < extra_samples else 0)
            
            if n_select == 3:
                # 如果需要选择三个样本，选择uncertainty最中等的
                idx1, idx2, idx3 = len(cluster_samples) // 4, len(cluster_samples) // 2, len(cluster_samples) * 3 // 4
                selected_samples.append(os.path.basename(cluster_samples[idx1][0]))
                selected_samples.append(os.path.basename(cluster_samples[idx2][0]))
                selected_samples.append(os.path.basename(cluster_samples[idx3][0])) 
            else:
                # 如果需要选择两个样本，选择uncertainty最大和最小的
                idx1, idx2 = len(cluster_samples) // 3, len(cluster_samples) * 2 // 3
                selected_samples.append(os.path.basename(cluster_samples[idx1][0]))  # 最小的uncertainty
                selected_samples.append(os.path.basename(cluster_samples[idx2][0]))  # 最大的uncertainty
        
        return selected_samples



    def get_parameters(self, select_ratio):
        """Get parameters needed for sample selection strategies."""
        results = []
        for filename in self.sample_list:
            file_path = os.path.join(self.base_dir, "images", self.split, filename)
            
            feature_path = file_path.replace("images/train", "feature").replace(".gz", ".npy")
            global_feature = np.load(feature_path)
            
            results.append([filename, global_feature])
            
        feats = np.array([result[1] for result in results])
        name_list = np.array([result[0] for result in results])
        num_samples = int(select_ratio * len(results))
        
        return feats, name_list, num_samples

    def save_samples(self, select_ratio, select_strategy):
        """Save selected samples to a text file."""
        save_dir = f"/root/autodl-fs/SAM_adaptive_learning/model/UTAH/train/{select_ratio}/{select_strategy}/2026"
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, "sample.txt")
        with open(save_path, "w") as f:
            for sample in self.final_sample_list:
                f.write(f"{sample}\n")
    
def get_data_loader(base_dir: str, split: str, select_ratio: float, select_strategy: str, transform: transforms.Compose):
    dataset = base_dir.split('/')[-1]
    assert dataset in ["Promise12", "UTAH"], f"dataset: {dataset} is not defined"
    if dataset == "Promise12":
        return Promise12(base_dir=base_dir, split=split, select_ratio=select_ratio, select_strategy=select_strategy, transform=transform)
    elif dataset == "UTAH":
        return UTAH(base_dir=base_dir, split=split, select_ratio=select_ratio, select_strategy=select_strategy, transform=transform)



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

