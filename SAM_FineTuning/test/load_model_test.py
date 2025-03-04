import sys
sys.path.append("..")
import torch
import config

vit_b_path = "/media/ubuntu/maxiaochuan/ISICDM 2024/SAM_FineTuning/checkpoints/sam_vit_b_01ec64.pth"
model = torch.load(vit_b_path)
with open("/media/ubuntu/maxiaochuan/ISICDM 2024/SAM_FineTuning/test/parameters.txt", 'w') as f:
    for param_name, parameter in model.items():
        f.write(f"{param_name}\n")
print(model)