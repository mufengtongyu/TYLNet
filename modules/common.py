# 文件: Improved-TCLNet - 副本/modules/common.py
import torch
import torch.nn as nn

class GSConv(nn.Module):
    """ GSConv (来自 TGE-YOLO 论文 [He et al., 2025] 的启发) """
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):
        super(GSConv, self).__init__()
        c_mid = c2 // 2
        # 深度可分离卷积 (DSC)
        self.dconv = nn.Conv2d(c1, c_mid, k, s, p, groups=c_mid, bias=False)
        self.bn_d = nn.BatchNorm2d(c_mid)
        # 标准卷积 (SC)
        self.sconv = nn.Conv2d(c1, c_mid, k, s, p, groups=g, bias=False)
        self.bn_s = nn.BatchNorm2d(c_mid)
        
        self.bn_out = nn.BatchNorm2d(c2)
        self.act = nn.ReLU() if act else nn.Identity()

    def forward(self, x):
        x_d = self.act(self.bn_d(self.dconv(x)))
        x_s = self.act(self.bn_s(self.sconv(x)))
        x_out = torch.cat([x_d, x_s], dim=1)
        # 注意：TGE-YOLO 中使用了 shuffle，
        # 但在轻量模型中，简单的 concat 已经足够高效
        return self.act(self.bn_out(x_out))

class CoordinateAttention(nn.Module):
    """ 坐标注意力 (CA) 模块 """
    def __init__(self, inp, oup, reduction=32):
        super(CoordinateAttention, self).__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))

        mip = max(8, inp // reduction)

        self.conv1 = nn.Conv2d(inp, mip, 1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = nn.Hardswish()

        self.conv_h = nn.Conv2d(mip, oup, 1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, oup, 1, stride=1, padding=0)

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

class GSBlock(nn.Module):
    """ 使用 GSConv 和 CA 替换原有的 ResBlock """
    def __init__(self, in_channel, out_channel, stride=1):
        super(GSBlock, self).__init__()
        self.conv1 = GSConv(in_channel, out_channel, k=3, s=stride, p=1)
        self.conv2 = GSConv(out_channel, out_channel, k=3, s=1, p=1)
        self.ca = CoordinateAttention(out_channel, out_channel)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channel != out_channel:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channel, out_channel, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channel)
            )
        self.relu = nn.ReLU()

    def forward(self, x):
        identity = self.shortcut(x)
        out = self.relu(self.conv1(x))
        out = self.conv2(out)
        out = self.ca(out) # 应用注意力
        out += identity # 残差连接
        return self.relu(out)
