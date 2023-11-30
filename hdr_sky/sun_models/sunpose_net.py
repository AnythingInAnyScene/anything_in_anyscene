import torch
import torchvision
import torch.nn as nn
import anything_in_anyscene.hdr_sky.utils as utils
import pdb

class sunposeLayer(nn.Module):
    def __init__(self, filter_in, filter_out, k_h=3, k_w=3, paddings=1, strides=1, dilation_rate=1):
        super(sunposeLayer, self).__init__()
        # self.conv1 = ops.conv2d(output_channels=filter_out, k_h=k_h, k_w=k_w, strides=strides)
        self.conv1 = nn.Conv2d(filter_in, filter_out, (k_h, k_w), stride=1, padding=paddings, dilation=1, groups=1, bias=True, padding_mode='zeros', device=None, dtype=None) 
        # self.norm1 = tfa.layers.InstanceNormalization()
        self.norm1 = nn.InstanceNorm2d(filter_out, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False, device=None, dtype=None)
        # self.actv1 = ops.relu()
        self.actv1 = nn.ReLU()
        
        # self.conv2 = ops.conv2d(output_channels=filter_out, k_h=k_h, k_w=k_w, strides=strides)
        self.conv2 = nn.Conv2d(filter_out, filter_out, (k_h, k_w), stride=1, padding=paddings, dilation=1, groups=1, bias=True, padding_mode='zeros', device=None, dtype=None) 
        # self.norm2 = tfa.layers.InstanceNormalization()
        self.norm2 = nn.InstanceNorm2d(filter_out, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False, device=None, dtype=None)
        # self.actv2 = ops.relu()
        self.actv2 = nn.ReLU()

    def forward(self, x):

        conv1 = self.conv1(x)
        norm1 = self.norm1(conv1)
        actv1  = self.actv1(norm1)

        conv2 = self.conv2(actv1)
        norm2 = self.norm2(conv2)
        actv2  = self.actv2(norm2)

        return actv2
    
class sunpose_model(nn.Module):
    def __init__(self, im_height=32, im_width= 128, da_kernel_size=3, dilation_rate=1):
        super(sunpose_model, self).__init__()

        self.fc_dim = int(im_height*im_width)

        # Sun position encoder
        self.sunlayer1 = sunposeLayer(3, 32, k_h=7, k_w=7, paddings=3)
        self.pool1_s  = nn.MaxPool2d(kernel_size = 2, stride = 2)

        self.sunlayer2 = sunposeLayer(32, 64, k_h=3, k_w=3, paddings=1)
        self.pool2_s  = nn.MaxPool2d(kernel_size = 2, stride = 2)

        self.sunlayer3 = sunposeLayer(64, 128, k_h=3, k_w=3, paddings=1)
        self.pool3_s  = nn.MaxPool2d(kernel_size = 2, stride = 2)

        # self.flat = tf.keras.layers.Flatten()
        # self.fc1 = tf.keras.layers.Dense(self.fc_dim)
        self.fc1 = nn.Linear(4*16*128, self.fc_dim)
        self.actv1_s  = nn.ReLU()
        # self.fc2 = tf.keras.layers.Dense(self.fc_dim)
        self.fc2 = nn.Linear(self.fc_dim, self.fc_dim)
        self.actv2_s  = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        # pdb.set_trace()
        sunlayer1 = self.sunlayer1(x)
        # print('sunlayer1 output: ', sunlayer1.size())
        pool1_s = self.pool1_s(sunlayer1)
        # print('pool1_s output: ', pool1_s.size())
        
        sunlayer2 = self.sunlayer2(pool1_s)
        # print('sunlayer2 output: ', sunlayer2.size())
        pool2_s = self.pool2_s(sunlayer2)
        # print('pool2_s output: ', pool2_s.size())

        sunlayer3 = self.sunlayer3(pool2_s)
        # print('sunlayer3 output: ', sunlayer3.size())
        pool3_s = self.pool3_s(sunlayer3)
        # print('pool3_s output: ', pool3_s.size())

        # flat = self.flat(pool3_s)
        pool3_s = pool3_s.reshape(pool3_s.size(0), -1)
        # print('pool3_s reshape output: ', pool3_s.size())
        fc1 = self.fc1(pool3_s)
        # print('fc1 output: ', fc1.size())
        actv1_s = self.actv1_s(fc1)
        # print('actv1_s output: ', actv1_s.size())
        fc2 = self.fc2(actv1_s)
        # print('fc2 output: ', fc2.size())
        actv2_s = self.actv2_s(fc2)        
        # print('actv2_s output: ', actv2_s.size())
        # sm = tf.nn.softmax(actv2_s)
        # sm = nn.Softmax(actv2_s)
        sm = self.softmax(actv2_s)
        # pdb.set_trace()
        
        # ## compute grad cam related operation inside forward
        # max_arg = torch.argmax(sun_poses, dim = 1)
        # max_arg = max_arg.unsqueeze(dim=-1) ### NOT SURE HERE, should this unsqueeze around the channel dim?? (should use 0 or -1)
        # y_c = torch.gather(sm, 1, max_arg)
        
        # sun_cam1 = utils.gradCamLayer(y_c, sunlayer1)
        # sun_cam2 = utils.gradCamLayer(y_c, sunlayer2)
        # sun_cam3 = utils.gradCamLayer(y_c, sunlayer3)
        
        return sm, [sunlayer1, sunlayer2, sunlayer3]