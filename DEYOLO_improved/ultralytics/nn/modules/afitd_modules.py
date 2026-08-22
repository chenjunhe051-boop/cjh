"""
AFITDYOLO Modules: MFFM, CAFM, MFEConv, C2f_MFE, ELA
Adapted for DEYOLO (RGB + IR/Depth dual-backbone) three-modality detection.
Paper: 基于YOLO的自适应多尺度红外目标检测网络 (光电工程 2026)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from .conv import Conv
from .block import Bottleneck


class ELA(nn.Module):
    def __init__(self, channels, kernel_size=7):
        super().__init__()
        self.pad = kernel_size // 2
        self.conv_h = nn.Conv1d(channels, channels, kernel_size=kernel_size, padding=self.pad, groups=channels, bias=False)
        self.conv_w = nn.Conv1d(channels, channels, kernel_size=kernel_size, padding=self.pad, groups=channels, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, h, w = x.shape
        x_h = x.permute(0, 3, 1, 2).contiguous().view(b * w, c, h)
        attn_h = self.conv_h(x_h).view(b, w, c, h).permute(0, 2, 3, 1)
        x_w = x.permute(0, 2, 1, 3).contiguous().view(b * h, c, w)
        attn_w = self.conv_w(x_w).view(b, h, c, w).permute(0, 2, 1, 3)
        return x * self.sigmoid(attn_h + attn_w)


class ShiftModule(nn.Module):
    def forward(self, x):
        b, c, h, w = x.shape
        x_r = torch.zeros_like(x); x_r[:, :, :, 1:] = x[:, :, :, :-1]
        x_l = torch.zeros_like(x); x_l[:, :, :, :-1] = x[:, :, :, 1:]
        x_d = torch.zeros_like(x); x_d[:, :, 1:, :] = x[:, :, :-1, :]
        x_u = torch.zeros_like(x); x_u[:, :, :-1, :] = x[:, :, 1:, :]
        return x + x_r + x_l + x_d + x_u


class FAM(nn.Module):
    """Full paper FAM with bmm spatial attention during training,
    lightweight 1x1 conv during eval/inference to avoid OOM."""
    def __init__(self, channels, lambda_init=0.5):
        super().__init__()
        self.channels = channels
        self.lambda_ = nn.Parameter(torch.tensor(lambda_init))
        self.conv_w1 = Conv(channels, channels, 3, act=False)
        self.conv_w2 = nn.Conv2d(channels, 1, 1, bias=False)
        self.sigmoid_w = nn.Sigmoid()
        self.conv_kv = nn.Conv2d(channels, channels, 1, bias=False)
        self.sigmoid_s = nn.Sigmoid()
        # Lightweight spatial attention for eval/inference
        self.conv_spatial = nn.Sequential(
            nn.Conv2d(channels, channels // 4, 1, bias=False),
            nn.BatchNorm2d(channels // 4),
            nn.SiLU(),
            nn.Conv2d(channels // 4, 1, 1, bias=False),
        )
        self.gamma = nn.Parameter(torch.ones(1))

    def forward(self, x, x_depth=None):
        b, c, h, w = x.shape
        w_feat = self.conv_w1(x_depth if x_depth is not None else x)
        W = self.sigmoid_w(self.conv_w2(w_feat))

        x_weighted = x * W
        x_inv_weighted = x * (1 - W)
        avg_p = F.adaptive_avg_pool2d(x_weighted, 1).view(b, c)
        max_p = torch.max(x_inv_weighted.view(b, c, -1), dim=2)[0]
        Ac = F.softmax(self.lambda_ * avg_p + (1 - self.lambda_) * max_p, dim=1).view(b, c, 1, 1)

        if self.training:
            # Full bmm spatial attention as in paper (training only, protected by checkpoint)
            kv = self.conv_kv(x)
            kv_flat = kv.view(b, c, h * w)
            attn_map = torch.bmm(kv_flat.transpose(1, 2), kv_flat) / (c ** 0.5)
            attn_map = attn_map.mean(dim=1, keepdim=True).view(b, 1, h, w)
            As = self.sigmoid_s(attn_map * W)
        else:
            # Lightweight 1x1 conv for eval/inference (no OOM)
            As = self.sigmoid_s(self.conv_spatial(x) * W)

        return self.gamma * (As * Ac * W) * x


class MFEConv(nn.Module):
    def __init__(self, c1, c2, k=3, shortcut=True):
        super().__init__()
        self.c_branch = c2 // 6
        self.shortcut = shortcut and c1 == c2
        self.shift = ShiftModule()
        self.conv_h1 = Conv(c1, self.c_branch, (1, k), 1, autopad((1, k)))
        self.conv_h2 = Conv(c1, self.c_branch, (1, k), 1, autopad((1, k)))
        self.conv_v1 = Conv(c1, self.c_branch, (k, 1), 1, autopad((k, 1)))
        self.conv_v2 = Conv(c1, self.c_branch, (k, 1), 1, autopad((k, 1)))
        self.conv_s1 = Conv(c1, self.c_branch, (k, k), 1, autopad((k, k)))
        self.conv_s2 = Conv(c1, self.c_branch, (k, k), 1, autopad((k, k)))
        self.fam = FAM(c2)
        self.conv1x1 = Conv(c2, c2, 1, act=False)

    def forward(self, x, x_depth=None):
        x_shift = self.shift(x)
        x3 = torch.cat([
            self.conv_h1(x_shift), self.conv_h2(x_shift),
            self.conv_v1(x_shift), self.conv_v2(x_shift),
            self.conv_s1(x_shift), self.conv_s2(x_shift)
        ], dim=1)
        # Gradient Checkpointing for training (saves memory during backward)
        if self.training:
            fam_out = checkpoint(self.fam, x3, x_depth, use_reentrant=False)
        else:
            fam_out = self.fam(x3, x_depth)
        return (x + self.conv1x1(x3 + fam_out)) if self.shortcut else self.conv1x1(x3 + fam_out)


def autopad(k, p=None, d=1):
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]
    return p


class C2f_MFE(nn.Module):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__()
        self.c = max((int(c2 * e) // 6) * 6, 6)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)
        self.m = nn.ModuleList(MFEConv(self.c, self.c, shortcut=shortcut) for _ in range(n))

    def forward(self, x):
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


class MFFM(nn.Module):
    def __init__(self, c1_shallow, c1_deep, c2, num_groups=4):
        super().__init__()
        self.c2 = c2
        self.num_groups = num_groups
        assert c2 % num_groups == 0
        self.g = c2 // num_groups
        self.conv_s = Conv(c1_shallow, c2, 1, 1)
        self.conv_d = Conv(c1_deep, c2, 1, 1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(c2, c2 // 16, bias=False), nn.ReLU(inplace=True),
            nn.Linear(c2 // 16, c2, bias=False), nn.Sigmoid()
        )
        self.sigmoid_s = nn.Sigmoid()
        self.group_convs = nn.ModuleList([
            nn.Sequential(nn.Conv2d(self.g, self.g, 1, bias=False), nn.BatchNorm2d(self.g))
            for _ in range(num_groups)
        ])
        self.bn_s = nn.BatchNorm2d(c2)
        self.bn_d = nn.BatchNorm2d(c2)
        self.sigmoid_spatial = nn.Sigmoid()
        self.ap_thresh = nn.AdaptiveAvgPool2d(1)
        self.ds_conv = nn.Sequential(
            nn.Conv2d(c2, c2, 3, 1, 1, groups=c2, bias=False), nn.BatchNorm2d(c2), nn.SiLU(),
            nn.Conv2d(c2, c2, 1, bias=False), nn.BatchNorm2d(c2),
        )
        self.gate_gen = nn.Sequential(nn.Conv2d(c2, c2, 1, bias=False), nn.BatchNorm2d(c2), nn.Sigmoid())
        self.conv_out3 = Conv(c2, c2, 1, act=False)
        self.ela = ELA(c2)
        self.final_conv = Conv(c2, c2, 1, act=False)

    def forward(self, x):
        x_shallow, x_deep = x
        b, _, h, w = x_shallow.shape
        xs = self.conv_s(x_shallow)
        xd = self.conv_d(x_deep)
        x_g = xs + xd

        y = self.avg_pool(x_g).view(b, self.c2)
        y1 = x_g * self.fc(y).view(b, self.c2, 1, 1)

        s = self.sigmoid_s(x_g)
        x_s = x_g * s
        x_groups = torch.split(x_s, self.g, dim=1)
        x_proc = [xg * F.softmax(self.group_convs[i](xg), dim=1) for i, xg in enumerate(x_groups)]
        y2 = torch.cat(x_proc, dim=1)

        ws = self.sigmoid_spatial(self.bn_s(xs))
        wd = self.sigmoid_spatial(self.bn_d(xd))
        w_thresh = self.sigmoid_spatial(self.ap_thresh(x_g))
        w_up_s = (ws > w_thresh).float() * ws
        w_low_s = (ws <= w_thresh).float() * ws
        w_up_d = (wd > w_thresh).float() * wd
        w_low_d = (wd <= w_thresh).float() * wd
        x_up = (w_up_s + w_up_d) * x_g
        x_low = (w_low_s + w_low_d) * x_g
        x_low_proc = self.ds_conv(x_low)
        y3 = self.conv_out3(x_up + x_low_proc * self.gate_gen(x_low_proc))

        out = y1 + y2 + y3
        out = self.ela(out)
        return self.final_conv(out)


class CAFM(nn.Module):
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
            x = x + 0.3 * x_ref_proj
        return x


class TriModalDEA(nn.Module):
    def __init__(self, channel=512, reduction=16):
        super().__init__()
        self.compress = Conv(channel * 3, channel, 3)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False), nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False), nn.Sigmoid()
        )
        self.conv_spatial = nn.Sequential(Conv(channel, channel // 2, 3), Conv(channel // 2, 3, 1, act=False))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x_rgb, x_ir, x_depth = x
        b, c, h, w = x_rgb.shape
        w_rgb = self.fc(self.avg_pool(x_rgb).view(b, c)).view(b, c, 1, 1)
        w_ir = self.fc(self.avg_pool(x_ir).view(b, c)).view(b, c, 1, 1)
        w_depth = self.fc(self.avg_pool(x_depth).view(b, c)).view(b, c, 1, 1)
        fused = self.compress(torch.cat([x_rgb, x_ir, x_depth], dim=1))
        spatial_w = self.sigmoid(self.conv_spatial(fused))
        w_r, w_i, w_d = spatial_w[:, 0:1], spatial_w[:, 1:2], spatial_w[:, 2:3]
        out = (x_rgb * w_rgb * w_r + x_ir * w_ir * w_i + x_depth * w_depth * w_d)
        return self.sigmoid(out + fused)