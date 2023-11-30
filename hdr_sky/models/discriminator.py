import numpy as np
import torch
import torchvision
import torch.nn as nn
import pdb
import sys
sys.path.append('../hdr_sky_cz')
from anything_in_anyscene.hdr_sky.utils import Conv2dSame, ResizeConv2d

class downsampling(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, strides=2, apply_norm=True):
        super(downsampling, self).__init__()
        # self.conv = tf.keras.layers.Conv2D(filters, kernel_size, strides=strides, padding="same", 
        #                                         kernel_initializer=tf.random_normal_initializer(0., 0.02),
        #                                         use_bias=False)
        self.conv = Conv2dSame(in_channels, out_channels, kernel_size, stride=strides, dilation=1, groups=1, bias=False)
        # CycleGAN way
        # self.norm = tfa.layers.InstanceNormalization()
        # self.norm = tf.keras.layers.BatchNormalization()
        self.norm = nn.BatchNorm2d(out_channels)
        # self.actv = tf.keras.layers.LeakyReLU()
        self.actv = nn.LeakyReLU(0.3)
        self.apply_norm = apply_norm

    def forward(self, x, training="training"):
        x = self.conv(x)
        if self.apply_norm:
            x = self.norm(x)
            # x = self.norm(x, training)
        x = self.actv(x)
        
        return x

class discriminatorModel(nn.Module):
    def __init__(self, im_height=32, im_width= 128, da_kernel_size=3, dilation_rate=1):
        super(discriminatorModel, self).__init__()

        # self.d1 = downsampling(64, 4, strides=2, apply_norm=False) # n,16,64,64
        self.d1 = downsampling(6, 64, (4,4), strides=2, apply_norm=False) # n,16,64,64
        # self.d2 = downsampling(128, 4, strides=2, apply_norm=True) # n,8,32,128
        self.d2 = downsampling(64, 128, (4,4), strides=2, apply_norm=True) # n,16,64,64
        # self.d3 = downsampling(256, 4, strides=2, apply_norm=True) # n,4,16,256
        self.d3 = downsampling(128, 256, (4,4), strides=2, apply_norm=True) # n,16,64,64
        # self.d4 = downsampling(512, 4, strides=1, apply_norm=True) # n,4,16,512
        self.d4 = downsampling(256, 512, (4,4), strides=1, apply_norm=True) # n,16,64,64
        
        # self.out = tf.keras.layers.Conv2D(1,4,strides=1,
        #                         kernel_initializer = tf.random_normal_initializer(0., 0.02))
        # self.out = nn.Conv2d(512, 1, (4, 4), stride=1, padding=1, dilation=1, groups=1, bias=False, padding_mode='zeros')  
        # self.out = Conv2dSame(512, 1, (4,4), stride=1, dilation=1, groups=1, bias=False)
        self.out = nn.Conv2d(512, 1, (4, 4), stride=1, padding=0, dilation=1, groups=1, bias=False, padding_mode='zeros') # this conv will change the shape of input feature
    
    def forward(self, x, training="training"):
        # x = tf.concat(x, axis=-1)
        x_cat = torch.cat(x, 1) # concatenate at channel
        x_cat = self.d1(x_cat, training)
        x_cat = self.d2(x_cat, training)
        x_cat = self.d3(x_cat, training)
        x_cat = self.d4(x_cat, training)

        x_cat = self.out(x_cat) # (batch, 1, 1, 13)
        # if use lsgan, do not use sigmoid
        return x_cat