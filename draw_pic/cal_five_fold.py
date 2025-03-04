import os
import re
import numpy as np
from collections import defaultdict
import nibabel as nib
from medpy.metric.binary import dc, hd95

def parse_model_path(path):
    """解析模型路径，返回(ratio, strategy, seed)"""
    pattern = r'train/([0-9.]+)/([^/]+)/(\d+)/unet_best_model.pth'
    match = re.match(pattern, path)
    if match:
        ratio, strategy, seed = match.groups()
        return float(ratio), strategy, int(seed)
    return None

def calculate_volume_metrics(pred_dir, label_dir):
    """
    计算一个完整volume的指标
    
    Args:
        pred_dir: 预测结果目录
        label_dir: 标签目录
    
    Returns:
        volumes_metrics: 每个volume的dice和hd95值的字典
    """
    volumes_metrics = {}
    volume_files = {}  # 用于存储每个volume的所有slice
    
    # 收集所有slice并按volume分组
    for filename in os.listdir(pred_dir):
        if filename.endswith('.nii.gz'):
            match = re.match(r'Case_(\d+)_slice_\d+\.nii\.gz', filename)
            if match:
                case_id = match.group(1)
                if case_id not in volume_files:
                    volume_files[case_id] = []
                volume_files[case_id].append(filename)
    
    # 对每个volume进行处理
    for case_id, slices in volume_files.items():
        # 按slice编号排序
        slices.sort(key=lambda x: int(re.search(r'slice_(\d+)', x).group(1)))
        
        # 读取并堆叠所有slice
        pred_volume = []
        label_volume = []
        
        for slice_file in slices:
            # 读取预测结果
            pred_path = os.path.join(pred_dir, slice_file)
            pred_slice = nib.load(pred_path).get_fdata()
            pred_volume.append(pred_slice)
            
            # 读取对应的标签
            label_path = os.path.join(label_dir, slice_file)
            label_slice = nib.load(label_path).get_fdata()
            label_volume.append(label_slice)
        
        # 转换为numpy数组
        pred_volume = np.stack(pred_volume, axis=0)
        label_volume = np.stack(label_volume, axis=0)
        
        # 计算指标
        dice_score = dc(pred_volume, label_volume)
        try:
            hd95_score = hd95(pred_volume, label_volume)
        except:
            hd95_score = 100.0  # 当计算失败时的默认值
        
        volumes_metrics[case_id] = {
            'dice': dice_score,
            'hd95': hd95_score
        }
    
    return volumes_metrics

def calculate_overall_stats(metrics_list):
    """
    计算多个volume的总体均值和标准差
    """
    metrics = np.array(metrics_list)
    mean = np.mean(metrics)
    std = np.std(metrics)
    return mean, std

def calculate_five_fold_metrics(log_file_path):
    # 用于存储每个(ratio, strategy)组合的结果
    results = defaultdict(lambda: defaultdict(list))
    
    # 读取日志文件
    with open(log_file_path, 'r') as f:
        for line in f:
            if 'Model:' not in line:
                continue
                
            # 解析每一行
            match = re.search(r'Model: (.*?), Dice Score: ([0-9.]+), Dice Std: ([0-9.]+)', line)
            if match:
                model_path, dice_score, dice_std = match.groups()
                parsed = parse_model_path(model_path)
                
                if parsed:
                    ratio, strategy, seed = parsed
                    results[ratio][strategy].append((float(dice_score), float(dice_std)))
    
    # 计算5折结果
    five_fold_results = []
    
    # 遍历每个ratio
    for ratio in sorted(results.keys()):
        strategies_with_seeds = defaultdict(list)
        
        # 对每个策略统计种子数量
        for strategy, scores in results[ratio].items():
            if len(scores) >= 3:  # 只处理至少有3个种子的策略
                strategies_with_seeds[len(scores)].append(strategy)
        
        # 分别处理3个种子和5个种子的策略
        for n_seeds, strategies in sorted(strategies_with_seeds.items()):
            print(f"\nRatio: {ratio}, Seeds: {n_seeds}")
            print("-" * 50)
            
            # 计算每个满足条件的策略的平均值和标准差
            for strategy in sorted(strategies):
                # 获取预测结果目录
                pred_dir = f"/root/autodl-tmp/SAM_adaptive_learning/predictions/{ratio}/{strategy}"
                label_dir = "/root/autodl-tmp/SAM_adaptive_learning/sam_data/UTAH/labels/valid"
                
                # 计算每个volume的指标
                volumes_metrics = calculate_volume_metrics(pred_dir, label_dir)
                
                # 提取所有volume的指标
                dice_scores = [m['dice'] for m in volumes_metrics.values()]
                hd95_scores = [m['hd95'] for m in volumes_metrics.values()]
                
                # 计算总体均值和标准差
                dice_mean, dice_std = calculate_overall_stats(dice_scores)
                hd95_mean, hd95_std = calculate_overall_stats(hd95_scores)
                
                five_fold_results.append({
                    'ratio': ratio,
                    'strategy': strategy,
                    'mean_dice': dice_mean,
                    'std_dice': dice_std,
                    'mean_hd95': hd95_mean,
                    'std_hd95': hd95_std,
                    'n_seeds': n_seeds
                })
                
                print(f"Strategy: {strategy:<30}")
                print(f"Dice Score: {dice_mean:.4f} ± {dice_std:.4f}")
                print(f"HD95 Score: {hd95_mean:.4f} ± {hd95_std:.4f}")
    
    return five_fold_results

if __name__ == "__main__":
    log_file_path = "/root/autodl-fs/SAM_adaptive_learning/model/UTAH/validation_results.txt"
    calculate_five_fold_metrics(log_file_path)