import sys
sys.path.append("/media/ubuntu/maxiaochuan/ISICDM 2024/SAM_FineTuning")
from segment_anything_finetune import sam_model_registry

def load_sam(
    model_type="vit_b",
    model_method=None,
             ):
    if model_type == "vit_b":
        sam_checkpoint = "/media/ubuntu/maxiaochuan/CLIP_SAM_zero_shot_segmentation/segment-anything/sam_vit_b_01ec64.pth" 
    elif model_type in ["vit_h", "auto_lora", "whole_box"]:
        sam_checkpoint = "/media/ubuntu//maxiaochuan/CLIP_SAM_zero_shot_segmentation/segment-anything/sam_vit_h_4b8939.pth"
    else: 
        sam_checkpoint = model_type
        model_type = model_method

    print(sam_checkpoint)
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam = sam.cuda()
   

            
 
    return sam

