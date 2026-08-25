"""
AFITDYOLO Modules v1.4 —— 只保留MFFM跨层融合

【v1.4】C2f_MFE改回C2f，只保留MFFM作为跨层融合模块
计算量与基线几乎相同，速度应该接近基线
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .conv import Conv


class MFFM(nn.Module):
    """v1.4精简版：只保留SE通道注意力 + 轻量空间门控 + 残差"""
    def __init__(self, c1_shallow, c1_deep, c2, num_groups=4):
        super().__init__()
        self.c2 = c2
        self.conv_s = Conv(c1_shallow, c2, 1, 1)
        self.conv_d = Conv(c1_deep, c2, 1, 1)

        # SE通道注意力
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.se = nn.Sequential(
            nn.Linear(c2, c2 // 16, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(c2 // 16, c2, bias=False),
            nn.Sigmoid()
        )

        # 轻量空间门控
        self.gate = nn.Sequential(
            nn.Conv2d(c2, c2, 3, 1, 1, groups=c2, bias=False),
            nn.BatchNorm2d(c2),
            nn.SiLU(),
            nn.Conv2d(c2, 1, 1, bias=False),
            nn.Sigmoid()
        )

        # 可学习残差权重
        self.res_weight = nn.Parameter(torch.tensor(0.5))

    def forward(self, x):
        x_shallow, x_deep = x
        b, c, h, w = x_shallow.shape

        # 1. 特征融合
        xs = self.conv_s(x_shallow)
        xd = self.conv_d(x_deep)
        x_g = xs + xd

        # 2. SE通道注意力
        se = self.se(self.avg_pool(x_g).view(b, self.c2)).view(b, self.c2, 1, 1)
        x_att = x_g * se

        # 3. 空间门控
        gate = self.gate(x_att)
        x_out = x_att * (1 + gate)

        # 4. 残差连接
        return x_g + self.res_weight * x_out


class CAFM(nn.Module):
    """保留代码但yaml中不再使用"""
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.mlp = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, 1, bias=False), nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, 1, bias=False)
        )
        self.sigmoid_c = nn.Sigmoid()
        self.conv_spatial = nn.Sequential(
            nn.Conv2d(2, 1, 7, padding=3, bias=False), nn.BatchNorm2d(1), nn.Sigmoid()
        )
        self.cross_proj = nn.Conv2d(channels, channels, 1, bias=False)
        self.bn_cross = nn.BatchNorm2d(channels)
        self.cross_gamma = nn.Parameter(torch.tensor(0.1))

    def forward(self, x, x_ref=None):
        avg_out = self.mlp(self.avg_pool(x))
        if x_ref is not None:
            ref_out = self.mlp(self.avg_pool(x_ref))
            channel_attn = self.sigmoid_c(avg_out + 0.5 * ref_out)
        else:
            channel_attn = self.sigmoid_c(avg_out)
        x = x * channel_attn

        avg_spatial = torch.mean(x, dim=1, keepdim=True)
        max_spatial, _ = torch.max(x, dim=1, keepdim=True)
        spatial_attn = self.conv_spatial(torch.cat([avg_spatial, max_spatial], dim=1))
        x = x * spatial_attn

        if x_ref is not None:
            x_ref_proj = F.silu(self.bn_cross(self.cross_proj(x_ref)))
            if x_ref_proj.shape[2:] != x.shape[2:]:
                x_ref_proj = F.interpolate(x_ref_proj, size=x.shape[2:], mode='bilinear', align_corners=False)
            x = x + self.cross_gamma * x_ref_proj
        return x