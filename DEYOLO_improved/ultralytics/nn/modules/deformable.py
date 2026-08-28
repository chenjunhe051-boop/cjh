# Ultralytics YOLO 🚀, AGPL-3.0 license
"""
Deformable & Multi-scale Fusion Modules
Based on: 融合多尺度特征的航拍目标检测算法 (Journal of System Simulation, 2025)
Add this file to: ultralytics/nn/modules/deformable.py
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .conv import Conv

# ------------------------------------------------------------------------------
# DCNv3
# ------------------------------------------------------------------------------
try:
    from torchvision.ops import deform_conv2d
    HAS_TORCHVISION_DCN = True
except ImportError:
    HAS_TORCHVISION_DCN = False
    deform_conv2d = None


class DCNv3(nn.Module):
    """
    Deformable Convolution v3 (simplified).
    Compatible with both old and new torchvision.
    """
    def __init__(self, c1, c2, k=3, s=1, p=None, g=1, act=True):
        super().__init__()
        if p is None:
            p = k // 2
        self.c1 = c1
        self.c2 = c2
        self.k = k
        self.s = s
        self.p = p
        self.g = g

        # offset (2*k*k) + mask (k*k) per group
        self.offset_mask_conv = nn.Conv2d(
            c1, g * (3 * k * k), k, s, p, groups=g, bias=True
        )
        self.weight = nn.Parameter(torch.empty(c2, c1 // g, k, k))
        self.bias = nn.Parameter(torch.empty(c2))
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())
        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)
        nn.init.constant_(self.offset_mask_conv.weight, 0)
        nn.init.constant_(self.offset_mask_conv.bias, 0)

    def forward(self, x):
        if not HAS_TORCHVISION_DCN:
            raise RuntimeError(
                "DCNv3 requires torchvision. Please install: pip install torchvision"
            )
        out = self.offset_mask_conv(x)
        o1, o2, mask = torch.chunk(out, 3, dim=1)
        offset = torch.cat([o1, o2], dim=1)
        mask = torch.sigmoid(mask)

        # 兼容旧版 torchvision（不支持 groups 参数）
        try:
            x = deform_conv2d(
                x, offset, self.weight, self.bias,
                stride=self.s, padding=self.p, groups=self.g, mask=mask
            )
        except TypeError:
            if self.g == 1:
                x = deform_conv2d(
                    x, offset, self.weight, self.bias,
                    stride=self.s, padding=self.p, mask=mask
                )
            else:
                x_split = x.split(self.c1 // self.g, dim=1)
                offset_split = offset.split(2 * self.k * self.k, dim=1)
                mask_split = mask.split(self.k * self.k, dim=1)
                out_list = []
                for xi, oi, mi, wi in zip(
                    x_split, offset_split, mask_split,
                    self.weight.split(self.c2 // self.g, dim=0)
                ):
                    bi = self.bias[
                        self.c2 // self.g * len(out_list):
                        self.c2 // self.g * (len(out_list) + 1)
                    ]
                    out_list.append(deform_conv2d(
                        xi, oi, wi, bi, stride=self.s, padding=self.p, mask=mi
                    ))
                x = torch.cat(out_list, dim=1)

        return self.act(self.bn(x))


# ------------------------------------------------------------------------------
# D2f  (Deformable C2f)
# ------------------------------------------------------------------------------
class D_Bottleneck(nn.Module):
    """Bottleneck with DCNv3 instead of standard Conv3x3."""
    def __init__(self, c1, c2, shortcut=True, g=1, e=0.5):
        super().__init__()
        c_ = int(c2 * e)
        self.dcv1 = DCNv3(c1, c_, 3, 1, g=g)
        self.dcv2 = DCNv3(c_, c2, 3, 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        return x + self.dcv2(self.dcv1(x)) if self.add else self.dcv2(self.dcv1(x))


class D2f(nn.Module):
    """
    Deformable C2f module. Same interface as C2f.
    Args: c1, c2, n=1, shortcut=False, g=1, e=0.5
    """
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__()
        self.c = int(c2 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)
        self.m = nn.ModuleList(D_Bottleneck(self.c, self.c, shortcut, g, e=1.0) for _ in range(n))

    def forward(self, x):
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


# ------------------------------------------------------------------------------
# GSConv  (Slim-neck)  —— 修复参数传递
# ------------------------------------------------------------------------------
class GSConv(nn.Module):
    """GSConv: Standard Conv + DWConv + ChannelShuffle."""
    def __init__(self, c1, c2, k=1, s=1, g=1, act=True):
        super().__init__()
        # ★ 修复：显式指定 g=，避免 g 被误传到 p 的位置
        self.cv1 = Conv(c1, c2 // 2, k, s, g=g, act=act)
        self.cv2 = Conv(c2 // 2, c2 // 2, 3, 1, g=c2 // 2, act=act)  # DWConv
        self.shuffle = nn.ChannelShuffle(2)

    def forward(self, x):
        x1 = self.cv1(x)
        x2 = self.cv2(x1)
        return self.shuffle(torch.cat([x1, x2], 1))


# ------------------------------------------------------------------------------
# RepBlock
# ------------------------------------------------------------------------------
class RepBlock(nn.Module):
    """RepBlock: 3x3 Conv -> 3x3 Conv with residual."""
    def __init__(self, c1, c2, k=3, s=1, p=1, act=True):
        super().__init__()
        self.conv1 = Conv(c1, c2, k, s, p, act=act)
        self.conv2 = Conv(c2, c2, k, s, p, act=act)
        self.add = c1 == c2 and s == 1

    def forward(self, x):
        y = self.conv1(x)
        y = self.conv2(y)
        return x + y if self.add else y


# ------------------------------------------------------------------------------
# FGM  (Feature Gather Module)
# ------------------------------------------------------------------------------
class FGM(nn.Module):
    """
    Feature Gather Module.
    Input : list of 4 tensors [f2, f3, f4, f5] (different scales from DEA)
    Output: single fused global feature map (target_size x target_size)
    """
    def __init__(self, c2, c3, c4, c5, out_c, target_size=40):
        super().__init__()
        self.target_size = target_size
        self.cv2 = Conv(c2, out_c // 4, 1, 1)
        self.cv3 = Conv(c3, out_c // 4, 1, 1)
        self.cv4 = Conv(c4, out_c // 4, 1, 1)
        self.cv5 = Conv(c5, out_c // 4, 1, 1)
        self.gs1 = GSConv(out_c, out_c // 2, 1, 1)
        self.rep = RepBlock(out_c // 2, out_c // 2, 3, 1, 1)
        self.gs2 = GSConv(out_c // 2, out_c, 1, 1)

    def forward(self, x):
        assert isinstance(x, list) and len(x) == 4, "FGM expects 4 input features [P2,P3,P4,P5]"
        f2, f3, f4, f5 = x
        f2 = F.adaptive_avg_pool2d(f2, self.target_size)
        f3 = F.adaptive_avg_pool2d(f3, self.target_size)
        f4 = F.interpolate(f4, size=(self.target_size, self.target_size), mode='bilinear', align_corners=False) \
            if f4.shape[-1] != self.target_size else f4
        f5 = F.interpolate(f5, size=(self.target_size, self.target_size), mode='bilinear', align_corners=False)
        fuse = torch.cat([self.cv2(f2), self.cv3(f3), self.cv4(f4), self.cv5(f5)], dim=1)
        fuse = self.gs2(self.rep(self.gs1(fuse)))
        return fuse


# ------------------------------------------------------------------------------
# IFM  (Information Fusion Module)
# ------------------------------------------------------------------------------
class IFM(nn.Module):
    """
    Information Fusion Module.
    Input : list of 2 tensors [local_feature, global_feature]
    Output: fused feature (same spatial size as local)
    """
    def __init__(self, c_local, c_global, out_c, act=True):
        super().__init__()
        self.conv_local = Conv(c_local, out_c, 1, 1)
        self.conv_act = nn.Sequential(
            Conv(c_global, out_c, 1, 1, act=False),
            nn.Sigmoid()
        )
        self.conv_embed = Conv(c_global, out_c, 1, 1, act=False)
        self.rep = RepBlock(out_c, out_c, 3, 1, 1, act=act)

    def forward(self, x):
        assert isinstance(x, list) and len(x) == 2, "IFM expects [local, global]"
        local, global_feat = x
        target_size = local.shape[2:]
        global_feat = F.interpolate(global_feat, size=target_size, mode='bilinear', align_corners=False)
        local = self.conv_local(local)
        act = self.conv_act(global_feat)
        embed = self.conv_embed(global_feat)
        fused = local * act + embed
        return self.rep(fused)