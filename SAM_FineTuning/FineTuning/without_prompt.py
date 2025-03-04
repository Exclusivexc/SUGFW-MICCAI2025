from segment_anything import sam_model_registry, SamPredictor
from segment_anything.utils.transforms import ResizeLongestSide


device = "cuda: 3"

# sam_checkpoint = "/media/ubuntu/maxiaochuan/CLIP_SAM_zero_shot_segmentation/segment-anything/sam_vit_b_01ec64.pth" 
# model_type = "vit_b"

sam_checkpoint = "/media/ubuntu//maxiaochuan/CLIP_SAM_zero_shot_segmentation/segment-anything/sam_vit_h_4b8939.pth"
model_type = "vit_h"

sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)
predictor = SamPredictor(sam)