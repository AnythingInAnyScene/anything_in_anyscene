import math
import torch
import torchvision.transforms as transforms

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def calc_psnr(pred, gt):
    diff = (pred - gt).pow(2).mean() + 1e-8
    return -10 * math.log10(diff)

def avg_psnr(output, gt, reduce=True):
    # PSNR should be calculated for each image, since sum(log) =/= log(sum).
    psnrs = []
    for b in range(gt.size(0)):
        psnr = calc_psnr(output[b], gt[b])
        psnrs.append(psnr)

    if reduce:
        return sum(psnrs) / len(psnrs)
    else:
        return sum(psnrs)

def save_fig(img_tensor, path):
    transform = transforms.ToPILImage()
    img_tensor = torch.clamp(img_tensor,min=0.0, max=1.0)
    img = transform(img_tensor)
    img.save(path)
