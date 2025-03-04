import os
import sys
import logging
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import argparse
from tqdm import tqdm
import glob
from collections import defaultdict
from medpy.metric import binary
import re

sys.path.append("/root/autodl-tmp/SAM_adaptive_learning")
from networks.net_factory import net_factory
from dataloaders.dataset import get_data_loader, ToTensor

def config():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_dir', type=str, default='')
    parser.add_argument('--dataset', type=str, default="Promise12")
    parser.add_argument('--model_root', type=str, default="/root/autodl-fs/SAM_adaptive_learning/model/Promise12", required=True, help='root directory containing model folders')
    parser.add_argument('--seed', type=int, default=2026)
    return parser.parse_args()

def collect_volume_slices(valloader, dataset):
    """将相同Case的切片收集到一起"""
    volumes = defaultdict(lambda: {'images': [], 'labels': [], 'slice_nums': []})
    
    for batch in valloader:
        image, label = batch['image'], batch['label']
        filenames = batch['filename']  # 假设dataloader返回文件名
        
        for img, lbl, fname in zip(image, label, filenames):
            # 解析文件名获取case编号和切片编号
            if dataset == "Promise12":
                match = re.match(r'Case(\d+)_slice_(\d+)\.nii\.gz', os.path.basename(fname))
            else:
                match = re.match(r'Case_(\d+)_(\d+)\.nii\.gz', os.path.basename(fname))
            if match:
                case_num, slice_num = map(int, match.groups())
                volumes[case_num]['images'].append(img)
                volumes[case_num]['labels'].append(lbl)
                volumes[case_num]['slice_nums'].append(slice_num)
    
    # 按切片编号排序
    
    for case_num in volumes:
        indices = np.argsort(volumes[case_num]['slice_nums'])
        volumes[case_num]['images'] = [volumes[case_num]['images'][i] for i in indices]
        volumes[case_num]['labels'] = [volumes[case_num]['labels'][i] for i in indices]

    
    return volumes

def validate_single_model(model_path, valloader, dataset):
    """验证单个模型"""
    model = net_factory(net_type='unet', in_chns=1, class_num=2)
    print(model_path)
    model.load_state_dict(torch.load("/root/autodl-tmp/SAM_adaptive_learning/draw_pic/iter_15400_dice_0.8333.pth"))
    model = model.cuda()
    model.eval()
    
    volumes = collect_volume_slices(valloader, dataset)
    dice_scores = []
    hd95_scores = []
    
    with torch.no_grad():
        for case_num, case_data in volumes.items():
            volume_preds = []
            volume_labels = []
            
            # 打印当前case的所有切片信息并记录顺序
            # logging.info(f"\nProcessing Case {case_num}")
            # logging.info("Slice order check:")
            
            # 创建排序索引
            sorted_indices = np.argsort(case_data['slice_nums'])
            sorted_slice_nums = [case_data['slice_nums'][i] for i in sorted_indices]
            
            # for slice_num in sorted_slice_nums:
                # logging.info(f"Processing slice {slice_num}")
            
            # 按排序后的索引处理每个切片
            for idx in sorted_indices:
                # 处理图像
                img = case_data['images'][idx]
                img = img.unsqueeze(0).cuda()
                output = model(img)
                output_soft = torch.softmax(output, dim=1)
                pred = output_soft.argmax(dim=1).cpu().numpy()[0]
                volume_preds.append(pred)
                
                # 处理标签
                lbl = case_data['labels'][idx]
                volume_labels.append(lbl.numpy())
            
            # 在堆叠之前输出检查信息
            # logging.info("Final slice order:")
            # for i, slice_num in enumerate(sorted_slice_nums):
                # logging.info(f"Position {i}: Slice {slice_num}")
            
            # 确认切片数量
            # logging.info(f"Number of prediction slices: {len(volume_preds)}")
            # logging.info(f"Number of label slices: {len(volume_labels)}")
            
            # 堆叠成volume
            volume_pred = np.stack(volume_preds, axis=0)
            volume_label = np.stack(volume_labels, axis=0)
            
            # 打印volume的形状
            # logging.info(f"Volume shapes - Pred: {volume_pred.shape}, Label: {volume_label.shape}")
            
            # 计算整个volume的Dice和HD95
            if volume_pred.sum():
                dice = binary.dc(volume_pred, volume_label)
                hd95 = binary.hd95(volume_pred, volume_label)
                
                dice_scores.append(dice)
                hd95_scores.append(hd95)
                
                # logging.info(f"Case {case_num}: Dice = {dice:.4f}, HD95 = {hd95:.4f}")
            else:
                pass
                # logging.info(f"Case {case_num}: Dice = 0, HD95 = 100")

    
    return np.mean(dice_scores), np.std(dice_scores), np.mean(hd95_scores), np.std(hd95_scores)

def main():
    args = config()
    
    # 设置日志
    log_path = os.path.join(args.model_root, 'validation_results.txt')
    logging.basicConfig(filename=log_path, level=logging.INFO,
                       format='[%(asctime)s.%(msecs)03d] %(message)s',
                       datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    
    # 加载验证数据集
    db_val = get_data_loader(
        base_dir=os.path.join("/root/autodl-tmp/SAM_adaptive_learning/sam_data", args.dataset),
        split="valid",
        select_ratio=None,
        select_strategy=None,
        transform=transforms.Compose([ToTensor()])
    )
    valloader = DataLoader(db_val, batch_size=1, shuffle=False, num_workers=1)
    
    # 查找所有包含best_model.pth的文件夹
    all_models = glob.glob(os.path.join(args.model_root, "**/*best_model.pth"), recursive=True)
    
    if not all_models:
        logging.error(f"No best_model.pth files found in {args.model_root}")
        return
    
    # 验证所有模型
    for model_path in tqdm(all_models, desc="Validating models"):
        dice_score, dice_std, hd95_score, hd95_std = validate_single_model(model_path, valloader, dataset=args.dataset)
        relative_path = os.path.relpath(model_path, args.model_root)
        logging.info(f"\nModel: {relative_path}")
        logging.info(f"Mean Volume Dice: {dice_score:.4f} (±{dice_std:.4f})")
        logging.info(f"Mean Volume HD95: {hd95_score:.4f} (±{hd95_std:.4f})")

if __name__ == "__main__":
    main()