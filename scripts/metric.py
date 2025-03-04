import numpy as np
import torch
from medpy.metric import dc, hd95
from scipy.ndimage import zoom

def get_metrics(image, mask, net, patch_size):
    image = image.squeeze().cpu().detach().numpy()
    mask = mask.squeeze(0).cpu().detach().numpy()
    
    x, y = image.shape
    image = zoom(image, (patch_size[0] / x, patch_size[1] / y), order=0)
    input = torch.from_numpy(image).unsqueeze(0).unsqueeze(0).float().cuda()
    net.eval()
    with torch.no_grad():
        out = torch.argmax(torch.softmax(net(input), dim=1), dim=1).squeeze(0)
        out = out.cpu().detach().numpy()
        pred = zoom(out, (x / patch_size[0], y / patch_size[1]), order=0)

    pred = pred.astype(np.float32)
    pred_positives = pred.sum()
    mask_positives = mask.sum()
    inter = (pred * mask).sum()
    union = pred_positives + mask_positives
    dice = (2 * inter) / (union + 1e-6)
    iou = inter / (union - inter + 1e-6)
    acc = (pred == mask).astype(np.float32).mean()
    recall = inter / (mask_positives + 1e-6)
    precision = inter / (pred_positives + 1e-6)
    f2 = (5 * inter) / (4 * mask_positives + pred_positives + 1e-6)
    mae = (np.abs(pred - mask)).mean()


    return [dice, iou, acc, recall, precision, f2, mae]