# import tensorflow as tf
# from tensorflow.keras import Model
# import ops
# import distortion_aware_ops as distortion_aware_ops
# import tensorflow_addons as tfa

# import tf_utils
import numpy as np
import torch
import torchvision
import torch.nn as nn
import pdb
import sys 
sys.path.append('../hdr_sky_cz')
# import hdr_sky_cz
from anything_in_anyscene.hdr_sky.utils import * #Conv2dSame, ResizeConv2d
from anything_in_anyscene.hdr_sky.models.sunrad_net import sunRadNet

class resBlock(nn.Module):
    def __init__(self, filter_in, filter_out, k_h=3, k_w=3, strides=1, dilation_rate=1):
        super(resBlock, self).__init__()
        # self.conv1 = ops.conv2d(output_channels=filter_out, k_h=k_h, k_w=k_w, strides=strides)
        self.conv1 = nn.Conv2d(filter_in, filter_out, (k_h, k_w), stride=1, padding=1, dilation=1, groups=1, bias=False, padding_mode='zeros') 
        # self.conv1 = distortion_aware_ops.conv2d(filter_out, kernel_size=k_h, strides=strides, dilation_rate=dilation_rate)
        # self.norm1 = tfa.layers.InstanceNormalization()
        self.norm1 = nn.InstanceNorm2d(filter_out, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False)
        
        # self.conv2 = ops.conv2d(output_channels=filter_out, k_h=k_h, k_w=k_w, strides=strides)
        self.conv2 = nn.Conv2d(filter_out, filter_out, (k_h, k_w), stride=1, padding=1, dilation=1, groups=1, bias=False, padding_mode='zeros')  
        # self.conv2 = distortion_aware_ops.conv2d(filter_out, kernel_size=k_h, strides=strides, dilation_rate=dilation_rate)
        # self.norm2 = tfa.layers.InstanceNormalization()
        self.norm2 = nn.InstanceNorm2d(filter_out, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False)
        
        if filter_in == filter_out:
            self.identity = lambda x : x
        else:
            # self.identity = ops.conv2d(filter_out, k_h=1, k_w=1, strides=1)
            self.identity = nn.Conv2d(filter_in, filter_out, (1, 1), stride=1, padding=0, dilation=1, groups=1, bias=False, padding_mode='zeros') 
        self.leak_relu = nn.LeakyReLU(0.3)

    def forward(self, x):

        conv1 = self.conv1(x)
        norm1 = self.norm1(conv1)
        # actv1  = tf.nn.leaky_relu(norm1, 0.1)
        actv1 = self.leak_relu(norm1)

        conv2 = self.conv2(actv1)
        norm2 = self.norm2(conv2)

        return torch.add(self.identity(x), norm2)

class resLayer(nn.Module):

    def __init__(self, filters, filter_in, k_h, k_w, strides=1, dilation_rate=1):
        super(resLayer, self).__init__()
        self.sequence = nn.ModuleList([])#list()

        for f_in, f_out in zip([filter_in]+ list(filters), filters):
            self.sequence.append(resBlock(f_in, f_out, k_h=k_h, k_w=k_w, strides=strides, dilation_rate=dilation_rate))
    
    def forward(self, x):
        for unit in self.sequence:
            x=unit(x)
        return x

class generatorModel(nn.Module):
    def __init__(self, device_list, batch_size = 32, im_height=32, im_width= 128, da_kernel_size=3, dilation_rate=1):
        super(generatorModel, self).__init__()

        self.fc_dim = int(im_height*im_width)
        self.im_height = im_height
        self.im_width = im_width
        self.thr = 0.12#torch.tensor(0.12)
        self.tensor_max_compare = torch.tensor(0.0).to(device_list)
        self.tensor_min_compare = torch.tensor(1.0).to(device_list)

        # sky encode fully conv layer
        # self.conv1_d = ops.conv2d(output_channels=32, k_h=7, k_w=7, strides=1) # TODO stride applied
        # self.conv1_d = nn.Conv2d(3, 32, (7, 7), stride=1, padding=3, dilation=1, groups=1, bias=False, padding_mode='zeros') # h,w remain the same
        self.conv1_d = Conv2dSame(3, 32, (7, 7), stride=1, dilation=1, groups=1, bias=False)

        # self.norm1_d = tfa.layers.InstanceNormalization()
        self.norm1_d = nn.InstanceNorm2d(32, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False)

        # self.conv2_d = ops.conv2d(output_channels=64, k_h=3, k_w=3, strides=2) # TODO stride applied
        # self.conv2_d = nn.Conv2d(32, 64, (3, 3), stride=2, padding=1, dilation=1, groups=1, bias=False, padding_mode='zeros') 
        self.conv2_d = Conv2dSame(32, 64, (3, 3), stride=2, dilation=1, groups=1, bias=False)

        # self.norm2_d = tfa.layers.InstanceNormalization()
        self.norm2_d = nn.InstanceNorm2d(64, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False)

        # self.conv3_d = ops.conv2d(output_channels=128, k_h=3, k_w=3, strides=2) # TODO stride applied
        # self.conv3_d = nn.Conv2d(64, 128, (3, 3), stride=2, padding=1, dilation=1, groups=1, bias=False, padding_mode='zeros')
        self.conv3_d = Conv2dSame(64, 128, (3, 3), stride=2, dilation=1, groups=1, bias=False) 
        # self.norm3_d = tfa.layers.InstanceNormalization()
        self.norm3_d = nn.InstanceNorm2d(128, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False)

        self.res = resLayer((128,128,128,128,128,128), 128, k_h=da_kernel_size, k_w=da_kernel_size, strides=1, dilation_rate=dilation_rate)

        # sky_decode
        # self.conv3_f = ops.deconv2d(output_channels=64, output_imshape=[int(im_height/ 2), int(im_width/ 2)], k_h=3, k_w=3, method='resize')
        self.conv3_f = ResizeConv2d(in_channels=128, out_channels=64, output_imshape=(16,64), k_h=3, k_w=3)
        # self.norm3_f = tfa.layers.InstanceNormalization()
        self.norm3_f = nn.InstanceNorm2d(64, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False)

        # self.conv2_f = ops.deconv2d(output_channels=32, output_imshape=[int(im_height), int(im_width)], k_h=3, k_w=3, method='resize')
        self.conv2_f = ResizeConv2d(in_channels=64, out_channels=32, output_imshape=(32,128), k_h=3, k_w=3)
        # self.norm2_f = tfa.layers.InstanceNormalization()
        self.norm2_f = nn.InstanceNorm2d(32, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False)

        # self.conv1_f = ops.conv2d(output_channels=3, k_h=7, k_w=7, strides=1)
        self.conv1_f = nn.Conv2d(32, 3, (7, 7), stride=1, padding=3, dilation=1, groups=1, bias=False, padding_mode='zeros')  

        # sun_decode
        # self.conv3_u = ops.deconv2d(output_channels=64, output_imshape=[int(im_height/ 2), int(im_width/ 2)], k_h=3, k_w=3, method='resize')
        self.conv3_u = ResizeConv2d(in_channels=128, out_channels=64, output_imshape=(16,64), k_h=3, k_w=3)
        # self.norm3_u = tfa.layers.InstanceNormalization()
        self.norm3_u = nn.InstanceNorm2d(64, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False)

        # self.conv2_u = ops.deconv2d(output_channels=32, output_imshape=[int(im_height), int(im_width)], k_h=3, k_w=3, method='resize')
        self.conv2_u = ResizeConv2d(in_channels=64, out_channels=32, output_imshape=(32,128), k_h=3, k_w=3)
        # self.norm2_u = tfa.layers.InstanceNormalization()
        self.norm2_u = nn.InstanceNorm2d(32, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False)

        # self.conv1_u = ops.conv2d(output_channels=3, k_h=7, k_w=7, strides=1)
        self.conv1_u = nn.Conv2d(32, 3, (7, 7), stride=1, padding=3, dilation=1, groups=1, bias=False, padding_mode='zeros')  

        # enhanceSunRadiance
        self.sun = sunRadNet(device_list)

        self.act_encode_1 = nn.LeakyReLU(0.1)
        self.act_encode_2 = nn.LeakyReLU(0.1)
        self.act_encode_3 = nn.LeakyReLU(0.1)
        self.act_skydecode_1 = nn.LeakyReLU(0.1)
        self.act_skydecode_2 = nn.LeakyReLU(0.1)
        self.act_skydecode_3 = nn.LeakyReLU(0.1)
        self.act_sundecode_1 = nn.LeakyReLU(0.1)
        self.act_sundecode_2 = nn.LeakyReLU(0.1)
        self.act_sundecode_3 = nn.LeakyReLU(0.1)

        self.relu_sky = nn.ReLU()
        self.relu_sun = nn.ReLU()

        self.resize = T.Resize((self.im_height, self.im_width)) # default BILINEAR method

    def encode(self, x, training="training"):
        
        conv1_d = self.conv1_d(x)
        norm1_d = self.norm1_d(conv1_d)
        # actv1_d = tf.nn.leaky_relu(norm1_d, 0.1)
        actv1_d = self.act_encode_1(norm1_d)

        conv2_d = self.conv2_d(actv1_d)
        norm2_d = self.norm2_d(conv2_d)
        # actv2_d = tf.nn.leaky_relu(norm2_d, 0.1)
        actv2_d = self.act_encode_2(norm2_d)

        conv3_d = self.conv3_d(actv2_d)
        norm3_d = self.norm3_d(conv3_d)
        # actv3_d = tf.nn.leaky_relu(norm3_d, 0.1)
        actv3_d = self.act_encode_3(norm3_d)

        out = self.res(actv3_d)

        return out

    def sky_decode(self, x, _input, training="training"):
        # pdb.set_trace()
        conv3_f = self.conv3_f(x)
        norm3_f = self.norm3_f(conv3_f)
        # actv3_f = tf.nn.leaky_relu(norm3_f, 0.1)
        actv3_f = self.act_skydecode_3(norm3_f)

        conv2_f = self.conv2_f(actv3_f)
        norm2_f = self.norm2_f(conv2_f)
        # actv2_f = tf.nn.leaky_relu(norm2_f, 0.1)
        actv2_f = self.act_skydecode_2(norm2_f)
        
        conv1_f = self.conv1_f(actv2_f)
        # sky_pred = tf.nn.leaky_relu(conv1_f, 0.1)
        sky_pred = self.act_skydecode_1(conv1_f)

        # sky_pred = tf.add_n([_input, sky_pred])
        sky_pred = _input + sky_pred
        # sky_pred = tf.nn.relu(sky_pred)
        sky_pred = self.relu_sky(sky_pred)
        return sky_pred
        
    def sun_decode(self, x, sun_cam1, sun_cam2, sun_cam3, sun_rad, training="training"):
        # pdb.set_trace()
        # In order to share the spatial information, I chose add gate for gradient distribution
        # sun_cam3_t = tf.tile(sun_cam3, [1,1,1,128])
        # skip3 = tf.add_n([sun_cam3_t, x])
        
        # conv3_u = self.conv3_u(skip3)
        conv3_u = self.conv3_u(x)
        norm3_u = self.norm3_u(conv3_u)
        # actv3_u = tf.nn.leaky_relu(norm3_u, 0.1)
        actv3_u = self.act_sundecode_3(norm3_u)

        # sun_cam2_t = tf.tile(sun_cam2, [1,1,1,64])
        # skip2 = tf.add_n([sun_cam2_t, actv3_u])

        # conv2_u = self.conv2_u(skip2)
        conv2_u = self.conv2_u(actv3_u)
        norm2_u = self.norm2_u(conv2_u)
        # actv2_u = tf.nn.leaky_relu(norm2_u, 0.1)
        actv2_u = self.act_sundecode_2(norm2_u)

        # sun_cam1_t = tf.tile(sun_cam1, [1,1,1,32])
        # skip1 = tf.add_n([sun_cam1_t, actv2_u])

        # conv1_u = self.conv1_u(skip1)
        conv1_u = self.conv1_u(actv2_u)
        # actv1_u = tf.nn.leaky_relu(conv1_u, 0.1)
        actv1_u = self.act_sundecode_1(conv1_u)
        
        # "add" preserves radiance value of the sun
        # sun_rad_t = tf.add_n([sun_rad, actv1_u])
        sun_rad_t = sun_rad + actv1_u
        # sun_rad_t = tf.nn.relu(sun_rad_t)
        sun_rad_t = self.relu_sun(sun_rad_t)
        return sun_rad_t

    def sun_rad_estimation(self, jpeg_img_float, sun_cam1, sun_cam2, sun_cam3, sunpose_pred, training="training"):
        # pdb.set_trace()
        # normed_sunpose_pred = tf.divide(sunpose_pred, tf.reduce_max(sunpose_pred))
        normed_sunpose_pred = torch.div(sunpose_pred, torch.max(sunpose_pred))
        # resized_sum_cam2 = tf.image.resize(sun_cam2, size=(self.im_height, self.im_width))
        # resized_sum_cam3 = tf.image.resize(sun_cam3, size=(self.im_height, self.im_width))
        resized_sum_cam2 = self.resize(sun_cam2)
        resized_sum_cam3 = self.resize(sun_cam3)
        
        # plz = tf.concat([jpeg_img_float, sun_cam1, resized_sum_cam2, resized_sum_cam3], axis=-1)
        plz = torch.cat((jpeg_img_float, sun_cam1, resized_sum_cam2, resized_sum_cam3), 1) # concatenate at channel
        sun_rad, gamma, beta = self.sun(normed_sunpose_pred, plz, training)

        # sun_rad_t = tf.tile(sun_rad, [1,1,1,3])
        sun_rad_t = torch.tile(sun_rad, [1,3,1,1]) # tile 1 channel to 3 channels

        return sun_rad_t, gamma, beta

    def blending(self, sky_pred, sun_pred, training="training"):
        # concat vs add  : add win!
        # out = tf.add_n([sky_pred, sun_pred])
        out = sky_pred + sun_pred

        return out
    
    def forward(self, jpeg_img_float, sunpose_pred, sun_cam1, sun_cam2, sun_cam3, training=True):
        # pdb.set_trace()
        res_out = self.encode(jpeg_img_float, training=training)
        sky_pred_gamma = self.sky_decode(res_out, jpeg_img_float, training=training)
        sky_pred_lin = hdr_logDecompression(sky_pred_gamma)

        # alpha = tf.reduce_max(sky_pred_lin, axis=[3])
        alpha, _ = torch.max(sky_pred_lin, 1) # reduce max along channel dimension
        # alpha = tf.minimum(1.0, tf.maximum(0.0, alpha - 1.0 + thr) / thr)
        # alpha = torch.minimum(1.0, torch.maximum(0.0, alpha - 1.0 + self.thr) / self.thr)
        
        # alpha = torch.minimum(torch.tensor(1.0), torch.maximum(torch.tensor(0.0), alpha - torch.tensor(1.0) + self.thr) / self.thr)
    
        alpha = torch.minimum(self.tensor_min_compare, torch.maximum(self.tensor_max_compare, alpha - 1.0 + 0.12) / 0.12)
        
        # alpha_c1 = tf.reshape(alpha, [-1, tf.shape(sky_pred_lin)[1], tf.shape(sky_pred_lin)[2], 1])
        alpha_c1 = torch.reshape(alpha, (-1, 1, self.im_height, self.im_width))
        # alpha_c3 = tf.tile(alpha_c1, [1, 1, 1, 3])
        alpha_c3 = torch.tile(alpha_c1, [1,3,1,1]) # tile 1 channel to 3 channels

        sun_rad_lin, gamma, beta = self.sun_rad_estimation(jpeg_img_float, sun_cam1, sun_cam2, sun_cam3, sunpose_pred, training= training)
        sun_rad_gamma = hdr_logCompression(sun_rad_lin)
        sun_pred_gamma = self.sun_decode(res_out, sun_cam1, sun_cam2, sun_cam3, sun_rad_gamma, training= training)

        # Rescaled sky_pred with alpha blending
        sky_pred_gamma = (1.- alpha_c3) * sky_pred_gamma
        sky_pred_lin = hdr_logDecompression(sky_pred_gamma)

        sun_pred_gamma = alpha_c3 * sun_pred_gamma
        sun_pred_lin = hdr_logDecompression(sun_pred_gamma)
        y_final_gamma = self.blending(sky_pred_gamma, sun_pred_gamma, training = training)
        y_final_lin = hdr_logDecompression(y_final_gamma)

        return [y_final_lin, y_final_gamma, sky_pred_lin, sun_pred_lin, gamma, beta, alpha_c3, sun_rad_lin]