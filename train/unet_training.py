import argparse
import logging
import os
import random
import shutil
import sys
sys.path.append("/root/autodl-tmp/SAM_adaptive_learning")
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.nn.modules.loss import CrossEntropyLoss
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
from dataloaders.dataset import get_data_loader, RandomCrop, RandomRotFlip, ToTensor
from networks.net_factory import net_factory
import torch.nn.functional as F
from PIL import Image
from networks.val_2D import test_single_volume


class DiceLoss(torch.nn.Module):
    def __init__(self, n_classes, onehot=True):
        super(DiceLoss, self).__init__()
        self.n_classes = n_classes
        self.onehot = onehot

    def _one_hot_encoder(self, input_tensor):
        tensor_list = []
        for i in range(self.n_classes):
            temp_prob = input_tensor == i * torch.ones_like(input_tensor)
            tensor_list.append(temp_prob)
        output_tensor = torch.cat(tensor_list, dim=1)
        return output_tensor.float()

    def _dice_loss(self, score, target):
        target = target.float()
        smooth = 1e-5
        intersect = torch.sum(score * target)
        y_sum = torch.sum(target * target)
        z_sum = torch.sum(score * score)
        loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
        loss = 1 - loss
        return loss

    def forward(self, inputs, target, weight=None, softmax=False):
        if softmax:
            inputs = torch.softmax(inputs, dim=1)
        if self.onehot:
            target = self._one_hot_encoder(target)
        if weight is None:
            weight = [1] * self.n_classes
        assert inputs.size() == target.size(), 'predict & target shape do not match'
        class_wise_dice = []
        loss = 0.0
        for i in range(0, self.n_classes):
            dice = self._dice_loss(inputs[:, i], target[:, i])
            class_wise_dice.append(1.0 - dice.item())
            loss += dice * weight[i]
        return loss / self.n_classes
    
def config():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--base_dir', type=str, default='/root/autodl-tmp/SAM_adaptive_learning/sam_data', help='base_dir of data')
    parser.add_argument('--dataset', type=str, default="Promise12", choices=["UTAH", "Prostate_ISBI2013", "BraTS2020", "Pancreas_MRI", "Promise12"])
    parser.add_argument('--exp', type=str, default="Fully", help="experiment_name")
    parser.add_argument('--model', type=str, default='unet', help='model_name')
    parser.add_argument('--save_model_name', type=str, default='unet')
    parser.add_argument('--max_epoch', type=int, default=400, help='maximum epoch number to train')
    parser.add_argument('--batch_size', type=int, default=16, help='batch_size per gpu')
    parser.add_argument('--deterministic', type=int,  default=1, help='whether use deterministic training')
    parser.add_argument('--base_lr', type=float,  default=0.01, help='segmentation network learning rate')
    parser.add_argument('--patch_size', nargs=2, type=int,  default=[480, 480], help='patch size of network input')
    parser.add_argument('--select_ratio', type=float,  default=0.01, help='select ratio')
    parser.add_argument('--select_strategy', type=str,  choices=["random", "FPS", "TypiClust", "CALR", "ALPS", "ProbCover", "my_min", "my_max", "my_middle", "my_greedy", "my_cluster_only", "my_uncertainty_only", "my_half_cluster", "my_half_cluster_far_uncertainty", "my_half_cluster_suitable_uncertainty"], default="random", help='select strategy')
    parser.add_argument('--seed', type=int,  default=2025, help='random seed')
    
    return parser.parse_args()



def train(args, snapshot_path):
    base_lr = args.base_lr
    base_dir = args.base_dir
    split = args.exp
    batch_size = args.batch_size
    dataset = args.dataset
    patch_size = args.patch_size
    select_ratio = args.select_ratio
    select_strategy=args.select_strategy
    class_num = 2
    model = net_factory(net_type=args.model, in_chns=1, class_num=class_num)

    dataset = args.dataset
    db_train = get_data_loader(base_dir=os.path.join(base_dir, dataset), 
                               split=split, 
                               select_ratio=select_ratio,
                               select_strategy=select_strategy,
                               transform=transforms.Compose([
                                   RandomCrop(patch_size),
                                   RandomRotFlip(),
                                   ToTensor(),
                               ]))
    db_val = get_data_loader(base_dir=os.path.join(base_dir, dataset), 
                             split="valid", 
                             select_ratio=select_ratio,
                               select_strategy=select_strategy,
                             transform=transforms.Compose([ToTensor()])
                             )
    
    
    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    trainloader = DataLoader(db_train, batch_size=batch_size, shuffle=True,
                             num_workers=4, pin_memory=True, worker_init_fn=worker_init_fn)
    valloader = DataLoader(db_val, batch_size=1, shuffle=False, num_workers=1)
    
    
    model.train()

    optimizer = optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)
    ce_loss = CrossEntropyLoss()
    dice_loss = DiceLoss(n_classes=class_num)

    logging.info("{} iterations per epoch".format(len(trainloader)))

    iter_num = 0
    max_iterations = 10000
    best_performance = 0.0
    # iterator = tqdm(range(args.max_epoch), ncols=70)
    for epoch_num in range(args.max_epoch):
        for i_batch, sampled_batch in enumerate(trainloader):
            image_batch, label_batch = sampled_batch['image'], sampled_batch['label']
            image_batch, label_batch = image_batch.cuda(), label_batch.cuda()
            outputs = model(image_batch)
            loss_ce = ce_loss(outputs, label_batch.long())
            
            # dice loss 
            outputs_soft = torch.softmax(outputs, dim=1)
            loss_dice = dice_loss(outputs_soft, label_batch.unsqueeze(1))
            loss = 0.5 * (loss_dice + loss_ce)
            # loss = loss_dice
            # loss = loss_ce
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_ 

            iter_num = iter_num + 1

            logging.info('iteration %d : loss : %f, loss_ce: %f, loss_dice: %f' % (iter_num, loss.item(), loss_ce.item(), loss_dice.item()))
            
            if iter_num > 0 and iter_num % 200 == 0: ## 200
                model.eval()
                dices = 0.0
                for i_batch, sampled_batch in enumerate(valloader):
                    image_batch, label_batch = sampled_batch['image'], sampled_batch['label']
                    image_batch, label_batch = image_batch.cuda(), label_batch.cuda()

                    with torch.no_grad():
                        outputs = model(image_batch)
                        # dice loss 
                        outputs_soft = torch.softmax(outputs, dim=1)
                        loss_dice = dice_loss(outputs_soft, label_batch.unsqueeze(1))
                                                
                        # metric_i = get_metrics(image_batch, label, model, patch_size=patch_size)
                        dices += (1 - loss_dice.item())
                dices = dices / len(db_val)

                performance = dices
                
                if performance > best_performance:
                    best_performance = performance
                    save_mode_path = os.path.join(snapshot_path, 'iter_{}_dice_{}.pth'.format(iter_num, round(best_performance, 4)))
                    save_best = os.path.join(snapshot_path, '{}_best_model.pth'.format(args.model))
                    torch.save(model.state_dict(), save_mode_path)
                    torch.save(model.state_dict(), save_best)

                logging.info(f'iteration {iter_num} : dice : {dices:.4f}')
                model.train()


            if iter_num >= max_iterations:
                break
        if iter_num >= max_iterations:
            # iterator.close()
            break
    return "Training Finished!"


if __name__ == "__main__":
    args = config()
    
    if not args.deterministic:
        cudnn.benchmark = True
        cudnn.deterministic = False
    else:
        cudnn.benchmark = False
        cudnn.deterministic = True

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    snapshot_path = f"/root/autodl-fs/SAM_adaptive_learning/model/{args.dataset}/{args.exp}/{args.select_ratio}/{args.select_strategy}/{args.seed}"
    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)
    
    logging.basicConfig(filename=snapshot_path+"/log.txt", level=logging.INFO, format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    train(args, snapshot_path)