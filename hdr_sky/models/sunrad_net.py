# import tensorflow as tf
# from tensorflow.keras.layers import Layer
# from tensorflow.keras import Model
# import tensorflow_addons as tfa
# import ops
import numpy as np
import torch
import torchvision
import torch.nn as nn
from anything_in_anyscene.hdr_sky.utils import * #Conv2dSame, ResizeConv2d
import pdb

class downsampling(nn.Module):
    # def __init__(self, filters, kernel_size, strides=2, apply_norm=True):
    def __init__(self, filter_in, filter_out, k_h=3, k_w=3, strides=2, apply_norm=True):    
        super(downsampling, self).__init__()
        # self.conv = tf.keras.layers.Conv2D(filters, kernel_size, strides=strides, padding="same", 
        #                                         kernel_initializer=tf.random_normal_initializer(0., 0.02),
        #                                         use_bias=False)
        # self.conv = nn.Conv2d(filter_in, filter_out, (k_h, k_w), stride=strides, padding=paddings, dilation=1, groups=1, bias=False, padding_mode='zeros', device=None, dtype=None)
        self.conv = Conv2dSame(filter_in, filter_out, (k_h, k_w), stride=strides, dilation=1, groups=1, bias=False)
        # CycleGAN way
        # # self.norm = tfa.layers.InstanceNormalization()
        # self.norm = tf.keras.layers.BatchNormalization()
        self.norm = nn.BatchNorm2d(filter_out)
        # self.actv = tf.keras.layers.LeakyReLU()
        self.actv = nn.LeakyReLU(0.3) # match tensorflow default negative slope 0.3
        self.apply_norm = apply_norm

    def forward(self, x, training="training"):
        x = self.conv(x)
        if self.apply_norm:
            x = self.norm(x)
            # x = self.norm(x, training)
        x = self.actv(x)
        
        return x

class sunRadNet(nn.Module):
    def __init__(self, device_list, epsilon=1e-5, pi=np.math.pi):
        super(sunRadNet, self).__init__()
        
        self.epsilon = epsilon
        # self.deltafunc_const = tf.sqrt(pi)
        self.deltafunc_const = torch.sqrt(torch.as_tensor(pi))

        # self.d1 = downsampling(64, 4, strides=2, apply_norm=False) # n,16,64,64
        self.d1 = downsampling(6, 64, 4, 4, strides=2, apply_norm=False) # n,16,64,64
        # self.d2 = downsampling(128, 4, strides=2, apply_norm=True) # n,8,32,128
        self.d2 = downsampling(64, 128, 4, 4, strides=2, apply_norm=True) # n,8,32,128
        # self.d3 = downsampling(256, 4, strides=2, apply_norm=True) # n,4,16,256
        self.d3 = downsampling(128, 256, 4, 4, strides=2, apply_norm=True) # n,4,16,256
        # # self.d4 = downsampling(512, 4, strides=1, apply_norm=True) # n,4,16,512
        # self.d4 = downsampling(256, 512, 3, 3, paddings=1, strides=1, apply_norm=True) # n,4,16,512
        self.d4 = downsampling(256, 512, 4, 4, strides=1, apply_norm=True) # n,4,16,512
        
        # self.flat = tf.keras.layers.Flatten()
        # self.gamma  = tf.keras.layers.Dense(1)
        self.gamma = nn.Linear(4*16*512, 1)
        # self.beta  = tf.keras.layers.Dense(1)
        self.beta  = nn.Linear(4*16*512, 1)
        self.sigmoid = nn.Sigmoid()
        self.tensor_where_compare = torch.tensor(30000., dtype=torch.float32).to(device_list)
        
    def forward(self, x, actv_map, training="training"):
        ## actv_map = tf.concat([jpeg_img_float, sun_cam1, resized_sum_cam2, resized_sum_cam3], axis=-1)
        # pdb.set_trace()
        d1 = self.d1(actv_map, training)
        d2 = self.d2(d1, training)
        d3 = self.d3(d2, training)
        d4 = self.d4(d3, training)

        # flat = self.flat(d4)
        flat = d4.reshape(d4.size(0), -1)
        gamma = self.gamma(flat)
        beta = self.beta(flat)

        gamma_in = self.sigmoid(gamma)
        gamma_in =  torch.reshape(gamma_in, [-1, 1, 1, 1])
        beta_in = self.sigmoid(beta)
        beta_in = torch.reshape(beta_in, [-1, 1, 1, 1])
        
        # Direc delta function (0 ~ infty)
        # # x = -tf.pow(tf.subtract(1., x), 2.)
        # x = torch.pow(torch.sub(1.0, x), 2.0)
        x = -torch.pow(1.0 - x, 2.0)
        # x = tf.divide(x, (beta_in + self.epsilon))
        div_x = torch.divide(x, (beta_in + self.epsilon))
        if torch.isnan(div_x).sum() > 0:
            pdb.set_trace()
            print("sunrad_net divide is nan")
        if torch.isinf(div_x).sum() > 0:
            pdb.set_trace()
            print("sunrad_net divide is nan")
        # x = tf.math.exp(x)
        # pdb.set_trace()
        exp_x = torch.exp(div_x)
        if torch.isnan(exp_x).sum() > 0:
            pdb.set_trace()
            print("sunrad_net exp is nan")
        if torch.isinf(exp_x).sum() > 0:
            pdb.set_trace()
            print("sunrad_net exp is nan")
        # x = tf.multiply(x, gamma_in)
        exp_x = exp_x * gamma_in
        # _const = tf.multiply(beta_in, self.deltafunc_const)
        _const = beta_in * self.deltafunc_const
        # x = tf.divide(x, (_const + self.epsilon))
        exp_x = torch.divide(exp_x, (_const + self.epsilon))
        # # x = tf.where(x > 30000., 30000. , x)
        # x = torch.where(x > 30000., torch.tensor(30000, dtype=x.dtype, device=x.device), x) # for numbers > 30000, change them to 30000, other numbers remain the same value
        exp_x = torch.where(exp_x > 30000., self.tensor_where_compare, exp_x) # for numbers > 30000, change them to 30000, other numbers remain the same value
        
        return exp_x, gamma_in, beta_in