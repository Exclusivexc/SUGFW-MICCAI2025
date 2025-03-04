import torch.nn as nn 
import torch.backends.cudnn as cudnn
import os
import numpy as np
import argparse
import random
import sys
import torch
from trainer import trainer

def config():
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--train_method', type=str, choices=["uncertainty_feature_prompt", "uncertainty_prompt", "auto_lora", "sam_lp", "prompt_lora", "prompt_new_lora"], required=True, help="train method")
    parser.add_argument('-cfp', '--choose_file_path', type=str, help="path of the iou score file ")
    parser.add_argument('-cm', '--choose_method', choices=["high", "low", "mean", "uniform", "random"], type=str, help="method to choose file")
    parser.add_argument('-cr', '--choose_ratio', default=0.01, type=str, help="choose ratio")
    parser.add_argument('-d', '--dataset', type=str, choices=["Promise12", "Pancreas_MRI", "UTAH"], required=True, help="dataset to train")
    parser.add_argument('-e', '--max_epoch', type=int, default=1000, help='maximum epoch number to train')
    parser.add_argument('-b', '--batch_size', type=int, default=1, help='batch_size per gpu')
    parser.add_argument('-lr', '--base_lr', type=float,  default=0.001, help='segmentation network learning rate')
    parser.add_argument('-p', '--patch_size', nargs=2, type=int,  default=[224, 224], help='patch size of network input')
    parser.add_argument('-s', '--seed', type=int,  default=2023, help='random seed')
    
    return parser.parse_args()

 
if __name__ == "__main__":
    args = config()
    
    cudnn.benchmark = False
    cudnn.deterministic = True

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    snapshot_path = f"./model/{args.dataset}/{args.train_method}/{os.path.splitext(args.choose_file_path)[0].split('/')[-1]}/{args.choose_method}/{args.choose_ratio}/{args.seed}"
    
    os.makedirs(snapshot_path, exist_ok=True)
    print(f"model will be saved to {os.path.abspath(snapshot_path)}!")
    
    trainer = trainer(
        snapshot_path=snapshot_path,
        train_method=args.train_method,
        choose_file_path=args.choose_file_path,
        choose_method=args.choose_method,
        choose_ratio=args.choose_ratio,
        dataset=args.dataset,
        max_epoch=args.max_epoch,
        batch_size=args.batch_size,
        base_lr=args.base_lr,
        patch_size=args.patch_size,
        seed=args.seed 
    ) 
    trainer.train()
    