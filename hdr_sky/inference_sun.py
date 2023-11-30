import torch
import torchvision
import torch.nn as nn
from anything_in_anyscene.hdr_sky.sun_models.sunpose_net import *
from anything_in_anyscene.hdr_sky.dataset import *
from pathlib import Path
import wandb
from tqdm import tqdm
import cv2
import pdb
from glob import glob
import matplotlib.pyplot as plt

import os 
os.environ['CUDA_VISIBLE_DEVICES'] = '4,5,6,7'
os.environ["OPENCV_IO_ENABLE_OPENEXR"]="1"

import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Type, Union, cast
import math
class Conv2dSame(nn.Conv2d):
    """
    Tensorflow like 'SAME' convolution wrapper for 2D convolutions.
    TODO: Replace with torch.nn.Conv2d when support for padding='same'
    is in stable version
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int, int]],
        stride: Union[int, Tuple[int, int]] = 1,
        padding: Union[int, Tuple[int, int]] = 0,
        dilation: Union[int, Tuple[int, int]] = 1,
        groups: int = 1,
        bias: bool = True,
    ) -> None:
        """
        See nn.Conv2d for more details on the possible arguments:
        https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html

        Args:

           in_channels (int): The expected number of channels in the input tensor.
           out_channels (int): The desired number of channels in the output tensor.
           kernel_size (int or tuple of int): The desired kernel size to use.
           stride (int or tuple of int, optional): The desired stride for the
               cross-correlation.
               Default: 1
           padding (int or tuple of int, optional): This value is always set to 0.
               Default: 0
           dilation (int or tuple of int, optional): The desired spacing between the
               kernel points.
               Default: 1
           groups (int, optional): Number of blocked connections from input channels
               to output channels. Both in_channels and out_channels must be divisable
               by groups.
               Default: 1
           bias (bool, optional): Whether or not to apply a learnable bias to the
               output.
        """
        super().__init__(
            in_channels, out_channels, kernel_size, stride, 0, dilation, groups, bias
        )

    def calc_same_pad(self, i: int, k: int, s: int, d: int) -> int:
        """
        Calculate the required padding for a dimension.

        Args:

            i (int): The specific size of the tensor dimension requiring padding.
            k (int): The size of the Conv2d weight dimension.
            s (int): The Conv2d stride value for the dimension.
            d (int): The Conv2d dilation value for the dimension.

        Returns:
            padding_vale (int): The calculated padding value.
        """
        return max((math.ceil(i / s) - 1) * s + (k - 1) * d + 1 - i, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:

            x (torch.tensor): The input tensor to apply 2D convolution to.

        Returns
            x (torch.Tensor): The input tensor after the 2D convolution was applied.
        """
        pdb.set_trace()
        ih, iw = x.size()[-2:]
        kh, kw = self.weight.size()[-2:]
        pad_h = self.calc_same_pad(i=ih, k=kh, s=self.stride[0], d=self.dilation[0])
        pad_w = self.calc_same_pad(i=iw, k=kw, s=self.stride[1], d=self.dilation[1])

        if pad_h > 0 or pad_w > 0:
            x = F.pad(
                x, [pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2]
            )
        return F.conv2d(
            x,
            self.weight,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )
        
    def pseudo_compute(self, x: torch.Tensor) -> torch.Tensor:
        pdb.set_trace()
        ih, iw = x.size()[-2:]
        print('ih, iw: ', ih, iw)
        kh, kw = self.weight.size()[-2:]
        print('kh, kw: ', kh, kw)
        pad_h = self.calc_same_pad(i=ih, k=kh, s=self.stride[0], d=self.dilation[0])
        pad_w = self.calc_same_pad(i=iw, k=kw, s=self.stride[1], d=self.dilation[1])
        print('pad_h, pad_w: ', pad_h, pad_w)
        print('self.dilation: ', self.dilation)
        print('self.padding: ', self.padding)
        print('self.groups: ', self.groups)
        
        print('x shape before pad: ', x.shape)
        if pad_h > 0 or pad_w > 0:
            x = F.pad(
                x, [pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2]
            )
        print('x shape after pad: ', x.shape)
        return F.conv2d(
            x,
            self.weight,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )
    
def inference(model, device):    
    ldr_transform = transforms.Compose([
        transforms.ToTensor()
    ])
    inference_img_dir = '/media/sandisk4T_2/lavel_sky_dataset/synthetic_data_test_small_entire_net/sun_mid_ldr/'
    inference_img_output_dir = '/media/sandisk4T_2/lavel_sky_dataset/pytorch_sunpose_net_inference_vis/synthetic_data/sun_mid_ldr/'
    
    if not os.path.isdir(inference_img_output_dir):
        os.mkdir(inference_img_output_dir)
    # hdr_imgs = [glob(os.path.join(inference_img_dir, '*.hdr'))]
    ldr_imgs = glob.glob(os.path.join(inference_img_dir, '*.jpg'))
    ldr_imgs = sorted(ldr_imgs)
    
    pdb.set_trace()
    
    for ldr_img_path in ldr_imgs:
        file_name = os.path.basename(ldr_img_path)
        file_name_split = os.path.splitext(file_name)
        # input bgr
        ldr_img = cv2.imread(ldr_img_path)
        ldr_img = ldr_img[:32, :128]
        h, w, _ = ldr_img.shape
        
        ldr_val = ldr_img / 255.0
        ldr_val = np.expand_dims(ldr_val, axis=0)
        ldr_val = ldr_transform(ldr_val)
        
        with torch.no_grad():
            ldr_val = ldr_val.to(device=device) 
            sm, sun_cam = inference(ldr_val)
            
        pred = torch.reshape(sm, (-1, 1, IMSHAPE[0], IMSHAPE[1]))
        # sungt = torch.reshape(sun_poses, (-1, 1, IMSHAPE[0], IMSHAPE[1]))
                    
                    
        # sun_cam2 = tf.image.resize(sun_cam[1], (IMSHAPE[0],IMSHAPE[1]))
        sun_cam2 = torch.reshape(sun_cam[1], (-1, 1, IMSHAPE[0], IMSHAPE[1]))
        sum_pred = sun_cam[0] * sun_cam2 * pred
        # sum_pred = sum_pred / (tf.reduce_max(sum_pred) + 1e-5)

        fig = plt.figure()

        ax = fig.add_subplot(6, 1, 1)
        ax.imshow(sun_cam[0][0])

        ax = fig.add_subplot(6, 1, 2)
        ax.imshow(sun_cam[1][0])

        ax = fig.add_subplot(6, 1, 3)
        ax.imshow(sun_cam[2][0])

        ax = fig.add_subplot(6, 1, 4)
        ax.imshow(pred[0])

        ax = fig.add_subplot(6, 1, 5)
        ax.imshow(sum_pred[0])

        ax = fig.add_subplot(6, 1, 6)
        ax.imshow(ldr_img)

        ax.set_title(ldr_img_path, fontsize=5)

        plt.savefig(inference_img_output_dir + "{}.png".format(hdr_file[0]))
        plt.clf()
        plt.close(fig)




if __name__ == "__main__":
    pdb.set_trace()
    conv_layer_s2_same = Conv2dSame(in_channels=3, out_channels=64, kernel_size=(3, 3), stride=(2, 2), groups=1, bias=True)
    # init model
    device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
    model = sunpose_model(im_height=32, im_width= 128)
    model.to(device)
    checkpoint = torch.load("/media/sandisk4T_2/lavel_sky_dataset/pytorch_checkpoints/SUN/laval_sky_epoch255.pth")
    model.load_state_dict(checkpoint)
    inference(model, device)