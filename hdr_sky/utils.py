# import tensorflow as tf
# import tensorflow_addons as tfa
import numpy as np
import pdb
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from typing import Dict, List, Optional, Tuple, Type, Union, cast
import math
PI = np.math.pi

def sphere2world(sunpose, h, w, skydome = True):
    x, y = sunpose
    
    # # unit_w = tf.divide(2 * PI, w)
    # unit_w = 2 * PI / w
    # # unit_h = tf.divide(PI, h * 2 if skydome else h)
    # if skydome == True:
    #     unit_h = PI / h * 2
    # else:
    #     unit_h = PI / h
    
    # # degree in xy coordinate to radian
    # theta = (x - 0.5 * w) * unit_w
    # # phi   = (h - y) * unit_h if skydome else (h * 0.5 - y) * unit_h
    # phi   = y * unit_h if skydome else y * unit_h

    # x_u = np.cos(phi) * np.cos(theta)
    # y_u = np.sin(phi)
    # z_u = np.cos(phi) * np.sin(theta)
    # p_u = [x_u, y_u, z_u]
    
    
    ### Note: changed this to maintain the same as dataset.py "def sunpose_init(i, h, w)"
    # xy coord to degree
    # gap value + init (half of the gap value)
    i = y * w + x
    x = ((i+1.) - np.floor(i/w) * w - 1.) * (360.0/w) + (360.0/(w*2.)) 
    y = (np.floor(i/w)) * (90./h) + (90./(2.*h))

    # deg2rad
    phi = (y) * (PI / 180.)
    theta = (x - 180.0) * (PI / 180.)

    # rad2xyz
    x_u = np.cos(phi) * np.cos(theta)
    y_u = np.sin(phi)
    z_u = np.cos(phi) * np.sin(theta)
    p_u = [x_u, y_u, z_u]
    
    # return np.array(p_u)

    return p_u

def gradCamLayer(y_c, A_k): ### NOT SURE HERE
    # pdb.set_trace()
    # # grad = tf.gradients(y_c, A_k)[0]
    # grad = torch.autograd.grad(y_c, A_k)[0]
    grad = torch.autograd.grad(y_c, A_k, grad_outputs=torch.ones_like(y_c),retain_graph=True)[0]

    # Global average pooling
    # weights = tf.reduce_mean(grad, axis = (1, 2))
    weights = torch.mean(grad, dim=(2,3)) # reduce mean of each image (height, width)
    # cam = tf.einsum('bc,bwhc->bwh', weights, A_k)
    cam = torch.einsum('bc,bcwh->bwh', weights, A_k) # note: pytorch channel locates at index=1
    
    # 계산된 weighted combination 에 ReLU 적용
    # cam = tf.nn.relu(cam)
    cam = torch.relu(cam)

    # get better result without normalization
    # cam = tf.divide(cam, (tf.reduce_max(cam) + 1e-10))
    
    # cam = tf.expand_dims(cam, axis=-1)
    cam = cam.unsqueeze(dim=1) # expand the channel dim

    return cam

def hdr_logCompression(x, validDR = 10.):
    # pdb.set_trace()
    # 0~1
    # disentangled way
    # x = tf.math.multiply(validDR, x)
    validDR = torch.tensor(validDR)
    x = validDR * x
    # numerator = tf.math.log(1.+ x)
    numerator = torch.log(1.+ x)
    # denominator = tf.math.log(1.+validDR)
    denominator = torch.log(1.+validDR)
    # output = tf.math.divide(numerator, denominator)
    output = torch.divide(numerator, denominator)

    return output

def hdr_logDecompression(x, validDR = 10.):
    # 0~1
    # denominator = tf.math.log(1.+validDR)
    validDR = torch.tensor(validDR)
    denominator = torch.log(1.+validDR)
    # x = tf.math.multiply(x, denominator)
    x = denominator * x
    # x = tf.math.exp(x)
    # pdb.set_trace()
    exp_x = torch.exp(x)
    if torch.isnan(exp_x).sum() > 0:
        pdb.set_trace()
        print("hdr_logDecompression exp is nan")
    if torch.isinf(exp_x).sum() > 0:
        pdb.set_trace()
        print("hdr_logDecompression exp is inf")
    # output = tf.math.divide(x-1., validDR)
    output = torch.divide(exp_x-1., validDR)
    return output



# def hdr_logCompression(x, validDR = 10.):
#     # 0~1
#     # disentangled way
#     # pdb.set_trace()
#     x = tf.math.multiply(validDR, x)
#     numerator = tf.math.log(1.+ x)
#     denominator = tf.math.log(1.+validDR)
#     output = tf.math.divide(numerator, denominator)

#     return output

# def hdr_logDecompression(x, validDR = 10.):
#     # 0~1
#     # pdb.set_trace()
#     denominator = tf.math.log(1.+validDR)
#     x = tf.math.multiply(x, denominator)
#     x = tf.math.exp(x)
#     output = tf.math.divide(x-1., validDR)
    
#     return output


# class Conv2dSame(nn.Conv2d):
#     """
#     Tensorflow like 'SAME' convolution wrapper for 2D convolutions.
#     TODO: Replace with torch.nn.Conv2d when support for padding='same'
#     is in stable version
#     """

#     def __init__(
#         self,
#         in_channels: int,
#         out_channels: int,
#         kernel_size: Union[int, Tuple[int, int]],
#         stride: Union[int, Tuple[int, int]] = 1,
#         padding: Union[int, Tuple[int, int]] = 0,
#         dilation: Union[int, Tuple[int, int]] = 1,
#         groups: int = 1,
#         bias: bool = True,
#     ) -> None:
#         """
#         See nn.Conv2d for more details on the possible arguments:
#         https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html

#         Args:

#            in_channels (int): The expected number of channels in the input tensor.
#            out_channels (int): The desired number of channels in the output tensor.
#            kernel_size (int or tuple of int): The desired kernel size to use.
#            stride (int or tuple of int, optional): The desired stride for the
#                cross-correlation.
#                Default: 1
#            padding (int or tuple of int, optional): This value is always set to 0.
#                Default: 0
#            dilation (int or tuple of int, optional): The desired spacing between the
#                kernel points.
#                Default: 1
#            groups (int, optional): Number of blocked connections from input channels
#                to output channels. Both in_channels and out_channels must be divisable
#                by groups.
#                Default: 1
#            bias (bool, optional): Whether or not to apply a learnable bias to the
#                output.
#         """
#         super().__init__(
#             in_channels, out_channels, kernel_size, stride, 0, dilation, groups, bias
#         )

#     def calc_same_pad(self, i: int, k: int, s: int, d: int) -> int:
#         """
#         Calculate the required padding for a dimension.

#         Args:

#             i (int): The specific size of the tensor dimension requiring padding.
#             k (int): The size of the Conv2d weight dimension.
#             s (int): The Conv2d stride value for the dimension.
#             d (int): The Conv2d dilation value for the dimension.

#         Returns:
#             padding_vale (int): The calculated padding value.
#         """
#         return max((math.ceil(i / s) - 1) * s + (k - 1) * d + 1 - i, 0)

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         """
#         Args:

#             x (torch.tensor): The input tensor to apply 2D convolution to.

#         Returns
#             x (torch.Tensor): The input tensor after the 2D convolution was applied.
#         """
#         ih, iw = x.size()[-2:]
#         kh, kw = self.weight.size()[-2:]
#         pad_h = self.calc_same_pad(i=ih, k=kh, s=self.stride[0], d=self.dilation[0])
#         pad_w = self.calc_same_pad(i=iw, k=kw, s=self.stride[1], d=self.dilation[1])

#         if pad_h > 0 or pad_w > 0:
#             x = F.pad(
#                 x, [pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2]
#             )
#         return F.conv2d(
#             x,
#             self.weight,
#             self.bias,
#             self.stride,
#             self.padding,
#             self.dilation,
#             self.groups,
#         )

class Conv2dSame(nn.Module):
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
        # super().__init__(
        #     in_channels, out_channels, kernel_size, stride, 0, dilation, groups, bias
        # )
        super(Conv2dSame, self).__init__()
        self.conv_layer = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=0, \
                                    dilation=dilation, groups=groups, bias=bias)
        self.kw = kernel_size[0]
        self.kh = kernel_size[1]
        self.stride = [stride, stride]
        self.dilation = [dilation, dilation]

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

    # def forward(self, x: torch.Tensor) -> torch.Tensor:
    def forward(self, x):
        """
        Args:

            x (torch.tensor): The input tensor to apply 2D convolution to.

        Returns
            x (torch.Tensor): The input tensor after the 2D convolution was applied.
        """
        # pdb.set_trace()
        ih, iw = x.size()[-2:]
        kh, kw = self.kh, self.kw#self.weight.size()[-2:]
        pad_h = self.calc_same_pad(i=ih, k=kh, s=self.stride[0], d=self.dilation[0])
        pad_w = self.calc_same_pad(i=iw, k=kw, s=self.stride[1], d=self.dilation[1])

        if pad_h > 0 or pad_w > 0:
            x = F.pad(
                x, [pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2]
            )
        output = self.conv_layer(x)
        return output

class ResizeConv2d(nn.Module):
    def __init__(self,
                 in_channels="input_channels",
                 out_channels="output_channels", 
                 output_imshape=(), # (height, width)
                 k_h=3, 
                 k_w=3,
                 strides=1,
                 padding=1):

        super(ResizeConv2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.k_w = k_w
        self.k_h = k_h
        self.strides = strides,
        self.output_imshape = output_imshape #tf.cast(output_imshape, dtype=tf.int32).numpy()
        self.padding = padding
        # self.method= method
        # self.kernel_initializer = kernel_initializer
        # self.bias_initializer = bias_initializer
        self.conv_layer = nn.Conv2d(self.in_channels, self.out_channels, (self.k_w,self.k_h), stride=self.strides, padding=self.padding, \
                                    dilation=1, bias=True) # set bias=True to match "add_bias" in tensorflow implementation
        self.resize = T.Resize(self.output_imshape) # default BILINEAR method


    def forward(self, x):
        # batch_size, input_h, _, _ = input.get_shape().as_list() #bhwc
        # im_resized = tf.image.resize(input, (self.output_imshape[0], self.output_imshape[1]), method=tf.image.ResizeMethod.BILINEAR)
        x_resized = self.resize(x)
        out = self.conv_layer(x_resized)
        # strides = [1,1,1,1]
        # deconv = tf.nn.bias_add(tf.nn.conv2d(im_resized, self.kernel, strides=strides, padding=self.padding), self.biases)
    
        return out
