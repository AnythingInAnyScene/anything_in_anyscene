import torch
import torch.nn as nn
import pdb

# #---- Group Norm Exp----#
# group_div = 1
# print("Group Norm: Divide group = ",group_div )
# #-----------------------

class MLP_act(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_list, act=nn.ReLU) -> None:
        super().__init__()
        layers = []
        lastv = in_dim
        for hidden in hidden_list:
            layers.append(nn.Linear(lastv, hidden))
            layers.append(act())
            lastv = hidden
        layers.append(nn.Linear(lastv, out_dim))
        self.layers = nn.Sequential(*layers)

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        x = self.layers(x)
        return x

class MLP(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_list, act=nn.ReLU) -> None:
        super().__init__()
        layers = []
        lastv = in_dim
        for hidden in hidden_list:
            layers.append(nn.Linear(lastv, hidden))
            lastv = hidden
        layers.append(nn.Linear(lastv, out_dim))
        layers.append(act())
        self.layers = nn.Sequential(*layers)

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        x = self.layers(x)
        return x

def MLP_with_EV(feature, step, base, mlp, EV_info=1, emb=None):
    if EV_info == 2:
        feature = feature.permute(0,2,3,1)
        w, h = feature.size()[1], feature.size()[2]
        EV_list = []

        for n in range(feature.size()[0]):
            n_step = torch.full((w, h, 1), step[n].item()).float().to(feature.device)
            n_origin = torch.full((w, h, 1), base[n].item()).float().to(feature.device)
            n_EV = torch.cat([n_step, n_origin], dim=-1)
            EV_list.append(n_EV)
        EV_all = torch.stack(EV_list)
        feature = torch.cat([feature, EV_all], dim=-1)

        feature = mlp(feature)
        feature = feature.permute(0,3,1,2)
        return feature

    elif EV_info == 1:
        feature = feature.permute(0,2,3,1)
        w, h = feature.size()[1], feature.size()[2]
        EV_list = []

        for n in range(feature.size()[0]):
            n_step = torch.full((w, h, 1), step[n].item()).float().to(feature.device)
            #n_origin = torch.full((w, h, 1), base[n].item()).float().to(feature.device)
            #n_EV = torch.cat([n_step, n_origin], dim=-1)
            EV_list.append(n_step)
        EV_all = torch.stack(EV_list)
        feature = torch.cat([feature, EV_all], dim=-1)

        feature = mlp(feature)
        feature = feature.permute(0,3,1,2)
        return feature

    elif EV_info == 3:
        feature = feature.permute(0,2,3,1)
        w, h = feature.size()[1], feature.size()[2]

        EV_list = []
        for n in range(feature.size()[0]):
            n_step = torch.full((w, h, 1), step[n].item()).float().to(feature.device)
            EV_list.append(n_step)
        EV_all = torch.stack(EV_list)# (bs, h, w, 1)
        EV_emb = emb(EV_all) #(bs, h, w, 16)
        feature = torch.cat([feature, EV_emb], dim=-1) #(bs, h, w, c+16)
        feature = mlp(feature) 
        feature = feature.permute(0,3,1,2)
        return feature

class Double_conv(nn.Module):
    def __init__(self, ch, act=nn.ReLU) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch, ch, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(ch, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            act(ch),

            nn.Conv2d(ch, ch, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(ch, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            act(ch)
        )

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        return x

class Bottel_neck(nn.Module):
    def __init__(self, in_ch, out_ch, h, w, mlp_num, down=None, residual=True, act=nn.ReLU, EV_info=1, emb=None, norm_type="LayerNorm",affine=True) -> None:
        super().__init__()
        self.hidden_dim = 256
        hidden_list = []
        for _ in range(mlp_num-1):
            hidden_list.append(self.hidden_dim)

        self.res = residual
        self.act = act
        self.EV_info = EV_info
        self.emb = emb
        self.affine = affine
        self.norm_type = norm_type

        if self.affine == False:
            print("BottleNeck normlization don't use affine(Learnable parameter)")

        print("#############################self.norm_type in bottleneck: ", self.norm_type, " ####################")
        self.downsample = down

        #---- Group Norm Exp----#
        #group_div = in_ch//2
        group_div = 2
        #-----------------------

        if self.norm_type == "LayerNorm":         
            #self.normalize = nn.LayerNorm([in_ch, h, w], elementwise_affine=self.affine)
            self.normalize = nn.GroupNorm(1,in_ch, affine=self.affine)
        elif self.norm_type == "GroupNorm":
            self.normalize = nn.GroupNorm(group_div,in_ch, affine=self.affine)
        elif self.norm_type == "InstanceNorm":
            self.normalize = nn.InstanceNorm2d(in_ch, affine=self.affine)

        if self.EV_info == 2:
            self.mlp = MLP(in_ch + 2, out_ch, hidden_list, act=self.act) 
        elif self.EV_info == 1:
            self.mlp = MLP(in_ch + 1, out_ch, hidden_list, act=self.act) 
        elif self.EV_info == 3:
            self.mlp = MLP(in_ch + 16, out_ch, hidden_list, act=self.act)

    def forward(self, x, s, b):
        if self.downsample != None:
            inputs = self.downsample(x)
        else:
            inputs = x

        if self.res:
            identity = inputs.clone()
            inputs = self.normalize(inputs)
            inputs = MLP_with_EV(inputs, s, b, self.mlp, EV_info=self.EV_info, emb=self.emb)
            return inputs + identity
        else:
            inputs = self.normalize(inputs)
            return MLP_with_EV(inputs, s, b, self.mlp, EV_info=self.EV_info, emb=self.emb)


class Upsample_MLP_multi_ResizeConvUp(nn.Module):
    def __init__(self, up_ch, in_ch, out_ch, h, w, mlp_num=3, residual=True, act=nn.ReLU, EV_info=1, emb=None, norm_type="LayerNorm", affine=True) -> None:
        super().__init__()
        self.res = residual
        self.act = act
        self.EV_info = EV_info
        self.emb = emb
        self.affine = affine
        self.norm_type = norm_type

        if self.affine == False:
            print("Decoder normlization don't use affine(Learnable parameter)")

        print("#############################self.norm_type in DecoderBlock: ", self.norm_type, " ####################")

        #---- Group Norm Exp----#
        #group_div = out_ch//2
        group_div = 2
        #-----------------------

        if self.norm_type == "LayerNorm":         
            #self.normalize = nn.LayerNorm([out_ch, h, w], elementwise_affine=self.affine)
            self.normalize = nn.GroupNorm(1,out_ch, affine=self.affine)
        elif self.norm_type == "GroupNorm":
            self.normalize = nn.GroupNorm(group_div,out_ch, affine=self.affine)
        elif self.norm_type == "InstanceNorm":
            self.normalize = nn.InstanceNorm2d(out_ch, affine=self.affine)

        hidden_list = []
        for _ in range(mlp_num-1):
            hidden_list.append(out_ch//2)

        #self.upconv = nn.ConvTranspose2d(up_ch, in_ch, kernel_size=2, stride=2)
        # Use resize + conv3x3
        mode = 'bicubic'
        print("Resize_conv upsample mode: ", mode)
        self.resize= nn.Upsample(scale_factor=2, mode=mode)
        self.conv_resize = nn.Conv2d(up_ch, in_ch, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))


        if self.EV_info == 2:
            self.mlp1 = MLP(in_ch *2 +2, out_ch, [], act=self.act)
            #self.normalize = nn.LayerNorm([out_ch, h, w], elementwise_affine=self.affine)
            self.mlp3 = MLP(out_ch +2, out_ch, hidden_list, act=self.act)
            self.mlp4 = MLP(out_ch +2, out_ch, hidden_list, act=self.act)
            
        elif self.EV_info == 1:
            self.mlp1 = MLP(in_ch *2 +1, out_ch, [], act=self.act)
            #self.normalize = nn.LayerNorm([out_ch, h, w], elementwise_affine=self.affine)
            self.mlp3 = MLP(out_ch +1, out_ch, hidden_list, act=self.act)
            self.mlp4 = MLP(out_ch +1, out_ch, hidden_list, act=self.act)

        elif self.EV_info == 3:
            self.mlp1 = MLP(in_ch *2 +16, out_ch, [], act=self.act)
            #self.normalize = nn.LayerNorm([out_ch, h, w], elementwise_affine=self.affine)
            self.mlp3 = MLP(out_ch +16, out_ch, hidden_list, act=self.act)
            self.mlp4 = MLP(out_ch +16, out_ch, hidden_list, act=self.act)        

    def forward(self, x:torch.Tensor, y:torch.Tensor, s:torch.Tensor, b:torch.Tensor) -> torch.Tensor:
        #x = self.upconv(x)
        x = self.resize(x)
        x = self.conv_resize(x)

        inputs = torch.cat([x, y], dim=1)
        inputs = MLP_with_EV(inputs, s, b, self.mlp1, EV_info=self.EV_info, emb=self.emb)

        if self.res:
            identity = inputs.clone()
            inputs = self.normalize(inputs)
            inputs = MLP_with_EV(inputs, s , b , self.mlp3, EV_info=self.EV_info, emb=self.emb)
            inputs = MLP_with_EV(inputs, s , b , self.mlp4, EV_info=self.EV_info, emb=self.emb)
            return inputs + identity
        else:
            inputs = self.normalize(inputs)
            inputs = MLP_with_EV(inputs, s , b , self.mlp3, EV_info=self.EV_info, emb=self.emb)
            return MLP_with_EV(inputs, s , b , self.mlp4, EV_info=self.EV_info, emb=self.emb)

class Upsample_MLP_multi_ResizeConvUp_Map(nn.Module):
    def __init__(self, up_ch, in_ch, out_ch, h, w, mlp_num=3, map_scale=2, residual=True, act=nn.ReLU, EV_info=1, emb=None, norm_type="LayerNorm", affine=True) -> None:
        super().__init__()
        self.res = residual
        self.act = act
        self.EV_info = EV_info
        self.emb = emb
        self.affine = affine
        self.norm_type = norm_type
        self.map_scale = map_scale

        if self.affine == False:
            print("Decoder normlization don't use affine(Learnable parameter)")

        print("#############################self.norm_type in DecoderBlock(Maps): ", self.norm_type, " ####################")

        #---- Group Norm Exp----#
        #group_div = out_ch//2
        group_div=2
        #-----------------------

        if self.norm_type== "LayerNorm":         
            #self.normalize = nn.LayerNorm([out_ch, h, w], elementwise_affine=self.affine)
            self.normalize = nn.GroupNorm(1,out_ch, affine=self.affine)
        elif self.norm_type== "GroupNorm":
            self.normalize = nn.GroupNorm(group_div,out_ch, affine=self.affine)
        elif self.norm_type== "InstanceNorm":
            self.normalize = nn.InstanceNorm2d(out_ch, affine=self.affine)

        hidden_list = []
        for _ in range(mlp_num-1):
            hidden_list.append(out_ch//2)

        #self.upconv = nn.ConvTranspose2d(up_ch, in_ch, kernel_size=2, stride=2)
        # Use resize + conv3x3
        mode = 'bicubic'
        print("Resize_conv upsample mode: ", mode)
        self.resize= nn.Upsample(scale_factor=2, mode=mode)
        self.conv_resize = nn.Conv2d(up_ch, in_ch, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))


        if self.EV_info == 2:
            self.mlp1 = MLP(in_ch *2 +2, out_ch, [], act=self.act)
            #self.ln = nn.LayerNorm([out_ch, h, w], elementwise_affine=self.affine)
            self.mlp3 = MLP(out_ch +2, out_ch, hidden_list, act=self.act)
            self.mlp4 = MLP(out_ch +2, out_ch, hidden_list, act=self.act)
            
        elif self.EV_info == 1:
            self.mlp1 = MLP(in_ch *2 +1, out_ch, [], act=self.act)
            #self.ln = nn.LayerNorm([out_ch, h, w], elementwise_affine=self.affine)
            self.mlp3 = MLP(out_ch +1, out_ch, hidden_list, act=self.act)
            self.mlp4 = MLP(out_ch +1, out_ch, hidden_list, act=self.act)

        elif self.EV_info == 3:
            self.mlp1 = MLP(in_ch *2 +16, out_ch, [], act=self.act)
            #self.ln = nn.LayerNorm([out_ch, h, w], elementwise_affine=self.affine)
            self.mlp3 = MLP(out_ch +16, out_ch, hidden_list, act=self.act)
            self.mlp4 = MLP(out_ch +16, out_ch, hidden_list, act=self.act)  

        if self.map_scale != 1:
            self.map_resize= nn.Upsample(scale_factor=self.map_scale, mode=mode)
            self.conv_rgb = nn.Conv2d(out_ch, 1, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))   
            self.conv_mul = nn.Conv2d(out_ch, 1, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))   

    def forward(self, x:torch.Tensor, y:torch.Tensor, s:torch.Tensor, b:torch.Tensor) -> torch.Tensor:
        #x = self.upconv(x)
        x = self.resize(x)
        x = self.conv_resize(x)

        inputs = torch.cat([x, y], dim=1)
        inputs = MLP_with_EV(inputs, s, b, self.mlp1, EV_info=self.EV_info, emb=self.emb)

        if self.res:
            identity = inputs.clone()
            inputs = self.normalize(inputs)
            inputs = MLP_with_EV(inputs, s , b , self.mlp3, EV_info=self.EV_info, emb=self.emb)
            inputs = MLP_with_EV(inputs, s , b , self.mlp4, EV_info=self.EV_info, emb=self.emb)
            feat_out = inputs + identity

            if self.map_scale != 1:
                #------ Output Map------
                feat_map = self.map_resize(feat_out)
                alpha = self.conv_mul(feat_map)
                beta = self.conv_rgb(feat_map)
                #-----------------------
                return feat_out, alpha, beta
            else:
                return feat_out
        else:
            inputs = self.normalize(inputs)
            inputs = MLP_with_EV(inputs, s , b , self.mlp3, EV_info=self.EV_info, emb=self.emb)
            return MLP_with_EV(inputs, s , b , self.mlp4, EV_info=self.EV_info, emb=self.emb)


def build_decoder(args):
    decode_name = args.decode_name

    implemented_decoder = ('mult_resizeUp', 'mult_resizeUp_map')
    assert decode_name  in implemented_decoder

    decoder = None

    if decode_name == 'mult_resizeUp':
        decoder = Upsample_MLP_multi_ResizeConvUp

    elif decode_name == 'mult_resizeUp_map':
        decoder = Upsample_MLP_multi_ResizeConvUp_Map

    return decoder
