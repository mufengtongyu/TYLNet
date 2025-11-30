from torch import nn
from torchvision.utils import save_image
import time
from datetime import datetime
from collections import OrderedDict
import torch
import os
import math

#################################################################
############ tools and components ###############################



#################################################################
############ 新增：h-sigmoid / h-swish 及 CA 模块 ################
class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6


class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x)


class CABlock(nn.Module):
    def __init__(self, channel, reduction=32):
        super(CABlock, self).__init__()

        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))

        mip = max(8, channel // reduction)

        self.conv1 = nn.Conv2d(channel, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = h_swish()

        self.conv_h = nn.Conv2d(mip, channel, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, channel, kernel_size=1, stride=1, padding=0)


    def forward(self, x):
        identity = x

        n, c, h, w = x.size()
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)

        y = torch.cat([x_h, x_w], dim=2)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y)

        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)

        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()

        out = identity * a_w * a_h

        return out


#################################################################
############ 新增：残差注意力包装器 ##############################
class ResidualAttention(nn.Module):
    def __init__(self, attention_module):
        super(ResidualAttention, self).__init__()
        self.attention = attention_module

    def forward(self, x):
        # 计算 x + Attention(x)␊
        return x + self.attention(x)

#################################################################


#################################################################
############ 新增：ECA 模块 ###################################

class ECABlock(nn.Module):
    def __init__(self, channel, b=1, gamma=2):
        super(ECABlock, self).__init__()
        kernel_size = int(abs((math.log(channel, 2) + b) / gamma))
        kernel_size = kernel_size if kernel_size % 2 else kernel_size + 1
        
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=kernel_size, padding=(kernel_size - 1) // 2, bias=False) 
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, h, w = x.size()
        # Squeeze
        y = self.avg_pool(x).squeeze(-1).squeeze(-1) # b, c
        # Excitation
        y = self.conv(y.unsqueeze(-1).transpose(-1, -2)).transpose(-1, -2).squeeze(-1) # b, c
        y = self.sigmoid(y).view(b, c, 1, 1) # b, c, 1, 1
        return x * y.expand_as(x)



#################################################################
############ 新增：SE 模块 ###################################

class SEBlock(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


#################################################################
############ 新增：SimAM 模块 ###################################
class SimAM(nn.Module):
    def __init__(self, e_lambda=1e-4):
        super(SimAM, self).__init__()
        self.activ = nn.Sigmoid()
        self.e_lambda = e_lambda

    def forward(self, x):
        b, c, h, w = x.size()
        
        # 空间维度上求均值和方差
        mu = torch.mean(x, dim=(2, 3), keepdim=True)
        var = torch.var(x, dim=(2, 3), keepdim=True)
        
        # 计算 e_t
        e_t = (x - mu) ** 2 / (4 * (var + self.e_lambda)) + 0.5
        
        # 能量函数
        return x * self.activ(e_t)


#################################################################
############ 新增：CBAM 模块 ###################################
class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc = nn.Sequential(nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False),
                               nn.ReLU(),
                               nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)

class CBAM(nn.Module):
    def __init__(self, channel, ratio=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.channel_attention = ChannelAttention(channel, ratio)
        self.spatial_attention = SpatialAttention(kernel_size)

    def forward(self, x):
        x = x * self.channel_attention(x)
        x = x * self.spatial_attention(x)
        return x

#################################################################
############ 原始模块 (保留) ####################################
class SKAttention(nn.Module):
    def __init__(self, channel=512, kernels=[1, 3, 5, 7], reduction=16, group=1, L=32):
        super().__init__()
        self.d = max(L, channel // reduction)
        self.convs = nn.ModuleList([])
        for k in kernels:
            self.convs.append(
            nn.Sequential(OrderedDict([
            ('conv', nn.Conv2d(channel, channel, kernel_size=k, padding=k // 2, groups=group)),
            ('bn', nn.BatchNorm2d(channel)),
            ('relu', nn.ReLU())
            ]))
            )
        self.fc = nn.Linear(channel, self.d)
        self.fcs = nn.ModuleList([])
        for i in range(len(kernels)):
            self.fcs.append(nn.Linear(self.d, channel))
            self.softmax = nn.Softmax(dim=0)

    def forward(self, x):
        bs, c, _, _ = x.size()
        conv_outs = []
        ### split
        for conv in self.convs:
            conv_outs.append(conv(x))
        feats = torch.stack(conv_outs, 0) # k,bs,channel,h,w

        ### fuse
        U = sum(conv_outs) # bs,c,h,w

        ### reduction channel
        S = U.mean(-1).mean(-1) # bs,c
        Z = self.fc(S) # bs,d

        ### calculate attention weight
        weights = []
        for fc in self.fcs:
            weight = fc(Z)
            weights.append(weight.view(bs, c, 1, 1)) # bs,channel
            attention_weights = torch.stack(weights, 0) # k,bs,channel,1,1
            attention_weights = self.softmax(attention_weights) # k,bs,channel,1,1

        ### fuse
        V = (attention_weights * feats).sum(0)
        return V


class ConvBlock(nn.Module):
    # a set of conv-bn-relu operation
    def __init__(self, inp_dim, out_dim, kernel_size, stride, use_bn, use_relu):
        super(ConvBlock, self).__init__()
        self.inp_dim = inp_dim
        self.conv = nn.Conv2d(inp_dim, out_dim, kernel_size, stride, padding=(kernel_size - 1) // 2, bias=True)
        self.relu = None
        self.bn = None
        if use_relu:
            self.relu = nn.ReLU(inplace=True)
        if use_bn:
            self.bn = nn.BatchNorm2d(out_dim)

    def forward(self, x):
        assert x.size()[1] == self.inp_dim, "{} {}".format(x.size(), self.inp_dim)
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x



class ResBlock(nn.Module):
    def __init__(self, in_dim,out_dim):
        super(ResBlock, self).__init__()

        self.relu = nn.ReLU()
        self.bn1 = nn.BatchNorm2d(in_dim)
        self.conv1 = ConvBlock(in_dim,int(out_dim/2),1,stride=1,use_bn=False,use_relu=False)
        self.bn2 = nn.BatchNorm2d(int(out_dim/2))
        self.conv2 = ConvBlock(int(out_dim/2),int(out_dim/2),3,stride=1,use_bn=False,use_relu=False)
        self.bn3 = nn.BatchNorm2d(int(out_dim/2))
        self.conv3 = ConvBlock(int(out_dim/2),out_dim,1,stride=1,use_bn=False,use_relu=False)
        self.expand_conv = None

        if in_dim != out_dim:
            self.expand_conv =ConvBlock(in_dim,out_dim,1,1,False,False)


    def forward(self, x):
        if self.expand_conv is not None:
            residual = self.expand_conv(x)
        else: residual = x

        out = self.bn1(x)
        out = self.relu(out)
        out = self.conv1(out)

        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)

        out = self.bn3(out)
        out = self.relu(out)
        out = self.conv3(out)
        out += residual
        return out



class net(nn.Module):
    def __init__(self, attention_type="se", use_skip_connection=False):
        super(net, self).__init__()
        self.feature_maps = {}
        self.use_skip_connection = use_skip_connection
        self.attention_type = attention_type.lower()

        self.maxpooling = nn.MaxPool2d(2,2)
        self.upsampling = nn.Upsample(scale_factor=2)

        self.pre = nn.Sequential( ConvBlock(6, 16, 7, 2, use_bn=True, use_relu=True),
                                  ConvBlock(16,32, 1,1,use_bn=False,use_relu=False),
                                  ResBlock(32,32),
                                  nn.MaxPool2d(2, 2),
                                  ResBlock(32,32),
                                  ConvBlock(32,64,1, 1, use_bn=False, use_relu=False),
                                  ResBlock(64,64))

        self.att0 = self._build_attention_block(channel=64)

        self.down1 = ResBlock(64,128)
        self.down2 = ResBlock(128,256)
        self.down3 = ResBlock(256,256)

        self.att1 = self._build_attention_block(channel=256)

        self.up3 = ResBlock(256,256)
        self.up2 = ResBlock(256,128)
        self.up1 = ResBlock(128,64)

        self.att2 = self._build_attention_block(channel=64)

        self.outter = nn.Sequential(ResBlock(64,64),
                                    ConvBlock(64, 64, 1, 1, use_bn=True, use_relu=True),
                                    ConvBlock(64, 1, 1, 1, use_bn=False, use_relu=False))

    def _build_attention_block(self, channel):
        if self.attention_type == 'baseline' or self.attention_type == 'baseline_skip':
            return nn.Identity()
        if self.attention_type == 'skattention':
            return SKAttention(channel=channel)
        if self.attention_type == 'cbam':
            return CBAM(channel=channel)
        if self.attention_type == 'resblock+cbam' or self.attention_type == 'resblock_cbam':
            return ResidualAttention(CBAM(channel=channel))
        if self.attention_type == 'ca':
            return CABlock(channel=channel)
        if self.attention_type == 'eca':
            return ECABlock(channel=channel)
        if self.attention_type == 'simam':
            return SimAM()
        # default to SE
        return SEBlock(channel=channel)

    def forward(self, x):
        x = self.pre(x)
        x = self.att0(x)
        if self.use_skip_connection:
            skip0 = x
        x = self.maxpooling(self.down1(x))
        if self.use_skip_connection:
            skip1 = x
        x = self.maxpooling(self.down2(x))
        if self.use_skip_connection:
            skip2 = x
        x = self.maxpooling(self.down3(x))
        x = self.att1(x)
        x = self.upsampling(self.up3(x))
        if self.use_skip_connection:
            x = x + skip2
        x = self.upsampling(self.up2(x))
        if self.use_skip_connection:
            x = x + skip1
        x = self.upsampling(self.up1(x))
        if self.use_skip_connection:
            x = x + skip0
        x = self.att2(x)
        x = self.outter(x)
        return x
    