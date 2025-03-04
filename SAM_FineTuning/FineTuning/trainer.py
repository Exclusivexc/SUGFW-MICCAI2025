import sys
sys.path.append("..")
sys.path.append("../..")
sys.path.append("../image_io")
sys.path.append("/media/ubuntu/maxiaochuan/SAM_adaptive_learning/scripts")
import logging
from torch.utils.data import DataLoader
import os
import random
from segment_anything_finetune import sam_model_registry
from segment_anything_finetune.utils.transforms import ResizeLongestSide
from dataloaders.dataloader import RandomCrop, RandomRotFlip, ToTensor, get_dataset
from torchvision import transforms
import torch.nn as nn 
import logging
import os
import numpy as np
import torch.optim as optim
import random
from tqdm import tqdm
from segment_anything_finetune.utils.transforms import ResizeLongestSide
from torchvision import transforms
import torch
from scipy.ndimage import zoom
from medpy.metric.binary import dc



class trainer():
    def __init__(
        self,  
        snapshot_path: str, 
        train_method: str="auto_lora",
        choose_file_path: str=None,
        choose_method: str="mean",
        choose_ratio: float=0.01,
        dataset: str="Promise12",
        max_epoch: int = 1000,
        batch_size: int = 1,
        base_lr: float = 0.002,
        patch_size: list = [224, 224],
        seed: int = 2023,
    ):
        self.snapshot_path = snapshot_path
        self.train_method = train_method
        self.choose_file_path = choose_file_path
        self.choose_method = choose_method
        self.choose_ratio = choose_ratio
        self.dataset = dataset
        self.max_epoch = max_epoch
        self.batch_size = batch_size
        self.base_lr = base_lr
        self.patch_size = patch_size
        self.seed = seed
        model_type_dict = {
            "auto_lora": "auto_lora",
            "prompt_lora_pad": "prompt_lora_pad",
            "pe_md": "vit_b",
            "pe": "vit_b",
            "md": "vit_b",
            "whole_box": "whole_box",
            "prompt_lora": "prompt_lora",
            "prompt_new_lora": "prompt_new_lora",
            "sam_lp": "sam_lp",
            "uncertainty_prompt": "uncertainty_prompt",
        }
        self.sam = self.load_sam(model_type=model_type_dict[train_method])
        self.sam_trans = ResizeLongestSide(self.sam.image_encoder.img_size)
        self.bce_loss = nn.BCELoss(reduction='mean')
        self.iter_num = 0
        self.max_iterations = 300000
        self.save_iterations = self.max_iterations // 100
        self.best_performance = 0
        # self.save_iterations = 2
        # 冻结部分参数后，使用优化器只更新需要训练的参数
        params_to_train = list(filter(lambda p: p.requires_grad, self.sam.parameters()))
            # 然后定义优化器

        self.optimizer = optim.SGD(params_to_train, lr=self.base_lr, momentum=0.9, weight_decay=0.0001)


        logging.basicConfig(filename=f"{self.snapshot_path}/log.txt", level=logging.INFO, format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%a %d %b %Y %H:%M:%S')

    def DiceLoss(self, preds, targets, smooth=1e-5):
        # 计算交集和并集
        intersection = torch.sum(preds * targets, dim=(2, 3))  # 交集
        union = torch.sum(preds * preds, dim=(2, 3)) + torch.sum(targets * targets, dim=(2, 3))  # 并集
        # 计算 Dice 系数
        dice = (2. * intersection + smooth) / (union + smooth)  # 加上平滑项避免除零

        dice = torch.sum(dice) / dice.shape[0]
        dice_loss = 1 - dice
        return dice_loss
        
    def load_sam(self, model_type="vit_b"):
        if model_type in ["vit_b", "auto_lora", "whole_box", "prompt_lora_pad", "prompt_lora", "prompt_new_lora", "sam_lp", "uncertainty_prompt"]:
            sam_checkpoint = "/root/autodl-tmp/SAM_adaptive_learning/SAM_FineTuning/checkpoints/sam_vit_b_01ec64.pth" 
        elif model_type in "vit_h":
            sam_checkpoint = "/media/ubuntu/maxiaochuan/SAM_adaptive_learning/SAM_FineTuning/checkpoints/sam_vit_h_4b8939.pth"
        else: 
            sam_checkpoint = model_type
            model_type = "vit_b"

        print(sam_checkpoint)
        sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
        sam = sam.cuda()
        with open("/root/autodl-tmp/SAM_adaptive_learning/SAM_FineTuning/test/parameters.txt", 'w') as f:
            for i, (param_name, param) in enumerate(sam.named_parameters()):
                # if "image_encoder" in param_name or "prompt_encoder" in param_name:
                param.requires_grad = False
                if self.train_method == "auto_lora":
                    if "lora" in param_name or "mask_decoder" in param_name: 
                        param.requires_grad = True
                if self.train_method == "prompt_lora_pad":
                    if "lora" in param_name or "prompt_encoder" in param_name or "mask_decoder" in param_name: 
                        param.requires_grad = True
                if self.train_method == "prompt_new_lora":
                    if "lora" in param_name or "prompt_encoder" in param_name or "mask_decoder" in param_name: 
                        param.requires_grad = True
                if self.train_method == "prompt_lora":
                    if "lora" in param_name or "prompt_encoder" in param_name or "mask_decoder" in param_name: 
                        param.requires_grad = True
                if self.train_method == "sam_lp":
                    if "lora" in param_name or "mask_decoder" in param_name or "prompt_encoder" in param_name: 
                        param.requires_grad = True
                if self.train_method == "uncertainty_prompt":
                    param.requires_grad = True
                    # if "lora" in param_name or "mask_decoder" in param_name or "prompt_encoder" in param_name: 
                    #     param.requires_grad = True


                
                f.write(f"{param_name}, {param.requires_grad} \n")
    
        return sam

    def get_box(self, H, W, label=None):
        d = random.randint(0, 50) if "pad" in self.train_method else 0
        if "auto" in self.train_method:
            box = None
        elif "whole" in self.train_method:
            box = [0, 0, H, W]
            box = np.array([box[1], box[0], box[3], box[2]])
            box = self.sam_trans.apply_boxes(box, original_size=(H, W))
            box = torch.Tensor(box).cuda()
        else: 
            box, labels = find_bounding_boxes_and_split_labels(label)
            box = box if box else [0, 0, H, W]
            for i in range(len(box)):
                box[i][0], box[i][1], box[i][2], box[i][3] = max(0, box[i][0] - d), max(0, box[i][1] - d), min(H, box[i][2] + d), min(W, box[i][3] + d)
            box = [[box[i][1], box[i][0], box[i][3], box[i][2]] for i in range(len(box))]
            box = np.array(box)
            box = self.sam_trans.apply_boxes(box, original_size=(H, W))
            box = torch.Tensor(box).cuda()
        return box, labels
    
    def val_one_with_box(self, image, label, filename):
        batched_input = []
        img = image.permute(1, 2, 0).detach().numpy()
        H, W = img.shape[:2]
        img = self.sam_trans.apply_image(img)
        img = torch.from_numpy(img).permute(2, 0, 1).cuda()
        box, labels = self.get_box(
            H=H, W=W,
            label=label
        )
        batched_input.append({
            'image': img,
            'original_size': (H, W),
            'boxes': box,
        })
        outputs = self.sam(
            batched_input=batched_input,
            multimask_output=False,
        )
        outputs = torch.cat([x["masks"] for x in outputs], dim=0)
        outputs = torch.sigmoid(outputs)
        out = torch.zeros_like(outputs).cuda()
        out[outputs > 0.5] = 1
        labels = torch.from_numpy(labels).cuda()
        print(out.shape, labels.shape)
            
        dice = 1 - self.DiceLoss(out, labels.unsqueeze(1))
        performance = dice.item()
        tqdm.write(f"{filename[0].split('/')[-1]} dice = {performance}")
        logging.info(f"{filename[0].split('/')[-1]} dice = {performance}")
        return performance

    def train_one_iter_with_box(self, image, label, optimizer):
        pass
    
    def train_one_iter_lp(self, image, label, uncertainty):
        batched_input = []
        H, W = self.patch_size
        for i in range(image.shape[0]):
            img = image[i].permute(1, 2, 0).detach().cpu().numpy()
            img = self.sam_trans.apply_image(img)
            img = torch.as_tensor(img, dtype=torch.float).permute(2, 0, 1).cuda()
            batched_input.append({
                'image': img,
                'original_size': (H, W),
                'uncertainty': uncertainty,
            })

        outputs = self.sam(
            batched_input=batched_input,
            multimask_output=False,
        )
        outputs = torch.cat([x["masks"] for x in outputs], dim=0)
        outputs = torch.sigmoid(outputs)
        loss_bce = self.bce_loss(outputs, label.unsqueeze(1))
        loss_dice = self.DiceLoss(outputs, label.unsqueeze(1))
        loss = 0.5 * loss_bce + 0.5 * loss_dice
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        self.iter_num += 1
        lr_ = self.base_lr * (1.0 - self.iter_num / self.max_iterations) ** 0.9
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr_ 

            
        logging.info('iteration %d : loss : %f, loss_bce: %f, loss_dice: %f, lr: %f' % (self.iter_num, loss.item(), loss_bce.item(), loss_dice.item(), lr_))
        tqdm.write('iteration %d : loss : %f, loss_bce: %f, loss_dice: %f, lr: %f' % (self.iter_num, loss.item(), loss_bce.item(), loss_dice.item(), lr_))
        

        return 
    
    def val_one_lp(self, image, label, filename, uncertainty):
        batched_input = []
        img = image.permute(1, 2, 0).detach().cpu().numpy()
        H, W = img.shape[:2]
        img = self.sam_trans.apply_image(img)
        img = torch.from_numpy(img).permute(2, 0, 1).cuda()
        batched_input.append({
            'image': img,
            'original_size': (H, W),
            'uncertainty': uncertainty,
        })
        outputs = self.sam(
            batched_input=batched_input,
            multimask_output=False,
        )
        outputs = torch.cat([x["masks"] for x in outputs], dim=0)
        outputs = torch.sigmoid(outputs)
        out = torch.zeros_like(outputs).cuda()
        out[outputs > 0.5] = 1
        print(out.shape, label.shape)
            
        dice = 1 - self.DiceLoss(out, label.unsqueeze(0).unsqueeze(0))
        performance = dice.item()
        tqdm.write(f"{filename[0].split('/')[-1]} dice = {performance}")
        logging.info(f"{filename[0].split('/')[-1]} dice = {performance}")
        return performance


        
            
    def train(self):
        def worker_init_fn(worker_id):
            random.seed(self.seed + worker_id)

        train_set = get_dataset(
            split="all",
            snapshot_path = self.snapshot_path,
            transform=transforms.Compose([
                RandomRotFlip(),
                RandomCrop(self.patch_size),
                ToTensor(),
            ])
        )
        val_set = get_dataset(
            split="valid",
            snapshot_path = self.snapshot_path,
            transform=ToTensor(),
        )
        train_loader = DataLoader(
            dataset=train_set,
            batch_size=self.batch_size,
            num_workers=4,
            worker_init_fn=worker_init_fn,
        )
        val_loader = DataLoader(
            dataset=val_set,
            batch_size=1,
            num_workers=4,
            worker_init_fn=worker_init_fn,
        )
        

        self.sam.train()
        logging.info(f"{self.max_iterations} iterations")
        print(f"{self.max_iterations} iterations")
        for _ in tqdm(range(self.max_epoch), ncols=100):
            for _, data_batch in enumerate(train_loader):
                image = data_batch["image"].cuda()
                label = data_batch["label"].cuda()
                uncertainty = data_batch["uncertainty"].squeeze().cuda()
                filename = data_batch["filename"]
                print(filename)
                self.train_one_iter_lp(image, label, uncertainty)
                
               
                if self.iter_num % self.save_iterations == 0: ## 200
                    metrics = 0.0
                    self.sam.eval()
                    for _, data_batch in enumerate(val_loader):
                        image, label, filename = data_batch["image"].squeeze().cuda(), data_batch["label"].squeeze().cuda(), data_batch["filename"]
                        uncertainty = data_batch["uncertainty"].squeeze().cuda()

                        with torch.no_grad():
                            performance = self.val_one_lp(image, label, filename, uncertainty)
                            metrics += performance
                        
                    metrics /= len(val_loader)
                    performance = metrics
                    if performance > self.best_performance:
                        self.best_performance = performance
                        save_mode_path = os.path.join(self.snapshot_path, f'iter_{self.iter_num}_dice_{round(self.best_performance, 4)}.pth')
                        save_best = os.path.join(self.snapshot_path, 'best_model.pth')
                        torch.save(self.sam.state_dict(), save_mode_path)
                        torch.save(self.sam.state_dict(), save_best)
                    

                    save_latest_path = os.path.join(self.snapshot_path, 'latest_model.pth')
                    torch.save(self.sam.state_dict(), save_latest_path)

                    logging.info('iteration %d : mean_dice : %f' % (self.iter_num, metrics))
                    tqdm.write('iteration %d : mean_dice : %f' % (self.iter_num, metrics))
                    self.sam.train()
            
                if self.iter_num >= self.max_iterations: 
                    break
            if self.iter_num >= self.max_iterations: 
                break


        print("Training Finished!")
        return
