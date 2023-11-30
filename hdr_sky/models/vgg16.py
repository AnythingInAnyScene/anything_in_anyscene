import numpy as np
# import torch
# import torchvision
# import torch.nn as nn
import pdb
import sys
sys.path.append('../hdr_sky_cz')

import torch
import torch.nn as nn
from torchvision.models.vgg import *
from torchvision.models.vgg import model_urls, cfgs
# from torchvision.models.utils import load_state_dict_from_url
# from torch.hub import load_state_dict_from_url
from torch.utils.model_zoo import load_url as load_state_dict_from_url
from typing import Type, Any, Callable, Union, List, Dict, Optional, cast
from torch import Tensor
from collections import OrderedDict


# class VGG16(nn.Module):
#     def __init__(self, num_classes=10):
#         super(VGG16, self).__init__()
#         self.layer1 = nn.Sequential(
#             nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
#             nn.BatchNorm2d(64),
#             nn.ReLU())
#         self.layer2 = nn.Sequential(
#             nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
#             nn.BatchNorm2d(64),
#             nn.ReLU(), 
#             nn.MaxPool2d(kernel_size = 2, stride = 2))
#         self.layer3 = nn.Sequential(
#             nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
#             nn.BatchNorm2d(128),
#             nn.ReLU())
#         self.layer4 = nn.Sequential(
#             nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
#             nn.BatchNorm2d(128),
#             nn.ReLU(),
#             nn.MaxPool2d(kernel_size = 2, stride = 2))
#         self.layer5 = nn.Sequential(
#             nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
#             nn.BatchNorm2d(256),
#             nn.ReLU())
#         self.layer6 = nn.Sequential(
#             nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
#             nn.BatchNorm2d(256),
#             nn.ReLU())
#         self.layer7 = nn.Sequential(
#             nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
#             nn.BatchNorm2d(256),
#             nn.ReLU(),
#             nn.MaxPool2d(kernel_size = 2, stride = 2))
#         self.layer8 = nn.Sequential(
#             nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
#             nn.BatchNorm2d(512),
#             nn.ReLU())
#         self.layer9 = nn.Sequential(
#             nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
#             nn.BatchNorm2d(512),
#             nn.ReLU())
#         self.layer10 = nn.Sequential(
#             nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
#             nn.BatchNorm2d(512),
#             nn.ReLU(),
#             nn.MaxPool2d(kernel_size = 2, stride = 2))
#         self.layer11 = nn.Sequential(
#             nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
#             nn.BatchNorm2d(512),
#             nn.ReLU())
#         self.layer12 = nn.Sequential(
#             nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
#             nn.BatchNorm2d(512),
#             nn.ReLU())
#         self.layer13 = nn.Sequential(
#             nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
#             nn.BatchNorm2d(512),
#             nn.ReLU(),
#             nn.MaxPool2d(kernel_size = 2, stride = 2))
#         self.fc = nn.Sequential(
#             nn.Dropout(0.5),
#             nn.Linear(7*7*512, 4096),
#             nn.ReLU())
#         self.fc1 = nn.Sequential(
#             nn.Dropout(0.5),
#             nn.Linear(4096, 4096),
#             nn.ReLU())
#         self.fc2= nn.Sequential(
#             nn.Linear(4096, num_classes))
        
#     def forward(self, x):
#         out = self.layer1(x)
#         out = self.layer2(out)
#         out = self.layer3(out)
#         out = self.layer4(out)
#         out = self.layer5(out)
#         out = self.layer6(out)
#         out = self.layer7(out)
#         out = self.layer8(out)
#         out = self.layer9(out)
#         out = self.layer10(out)
#         out = self.layer11(out)
#         out = self.layer12(out)
#         out = self.layer13(out)
#         out = out.reshape(out.size(0), -1)
#         out = self.fc(out)
#         out = self.fc1(out)
#         out = self.fc2(out)
#         return out

class newVGG(VGG):
    def __init__(self,
                 features: nn.Module,
                 **kwargs: Any) -> None:
        super().__init__(features,**kwargs)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        return x

class NewModel(nn.Module):

    def __init__(self,
                 base_model: str,
                 batch_norm: bool,
                 base_out_layer: int,
                 num_trainable_layers: int) -> None:

        super().__init__()
        self.base_model = base_model
        self.batch_norm = batch_norm
        self.base_out_layer = base_out_layer
        self.num_trainable_layers = num_trainable_layers
        

        self.cfg_dict = {'vgg11':'A',
                         'vgg13':'B',
                         'vgg16':'D',
                         'vgg19':'E'}

        self.vgg = self._vgg(self.base_model,
                             self.cfg_dict[self.base_model],
                             self.batch_norm,
                             self.base_out_layer,
                             True, True)

        
        self.total_children = 0
        self.children_counter = 0
        for c in self.vgg.children():
            self.total_children += 1
            
        if self.num_trainable_layers == -1:
            self.num_trainable_layers = self.total_children
            
        for c in self.vgg.children():
            if self.children_counter < self.total_children - self.num_trainable_layers:
                for param in c.parameters():
                    param.requires_grad = False
            else:
                for param in c.parameters():
                    param.requires_grad = True
            self.children_counter += 1

    def make_layers(self,
                    cfg: List[Union[str, int]],
                    base_out_layer: int, 
                    batch_norm: bool = False) -> nn.Sequential:
        layers: List[nn.Module] = []
        in_channels = 3
        layer_count = 0
        for v in cfg:
            if v == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                v = cast(int, v)
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
                if batch_norm:
                    layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
                else:
                    layers += [conv2d, nn.ReLU(inplace=True)]
                in_channels = v
                
                # layer_count += 1
            layer_count += 1
            if layer_count == base_out_layer:
                break

        return nn.Sequential(*layers)

    def _vgg(self, 
            arch: str, 
            cfg: str, 
            batch_norm: bool, 
            base_out_layer: int,
            pretrained: bool, 
            progress: bool, 
            **kwargs: Any) -> newVGG:

        model = newVGG(self.make_layers(cfgs[cfg], 
                                   base_out_layer, 
                                   batch_norm=batch_norm), 
                       **kwargs)
        # pdb.set_trace()
        if pretrained:
            state_dict = load_state_dict_from_url(model_urls[arch],
                                                progress=progress)
            model.load_state_dict(state_dict, strict = False)
        return model
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.vgg(x)
        return x