"""
DEA (DECA and DEPA) module + DEA3 three-modal fusion
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .conv import Conv


class DEA(nn.Module):
    """x0 --> RGB feature map,  x1 --> IR feature map"""

    def __init__(self, channel=512, kernel_size=80, p_kernel=None, m_kernel=None, reduction=16):
        super().__init__()
        self.deca = DECA(channel, kernel_size, p_kernel, reduction)
        self.depa = DEPA(channel, m_kernel)
        self.act = nn.Sigmoid()

    def forward(self, x):
        result_vi, result_ir = self.depa(self.deca(x))
        return self.act(result_vi + result_ir)


class DECA(nn.Module):
    """x0 --> RGB feature map,  x1 --> IR feature map"""

    def __init__(self, channel=512, kernel_size=80, p_kernel=None, reduction=16):
        super().__init__()
        self.kernel_size = kernel_size
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )
        self.act = nn.Sigmoid()
        self.compress = Conv(channel * 2, channel, 3)

        """convolution pyramid"""
        if p_kernel is None:
            p_kernel = [5, 4]
        kernel1, kernel2 = p_kernel
        self.conv_c1 = nn.Sequential(nn.Conv2d(channel, channel, kernel1, kernel1, 0, groups=channel), nn.SiLU())
        self.conv_c2 = nn.Sequential(nn.Conv2d(channel, channel, kernel2, kernel2, 0, groups=channel), nn.SiLU())
        self.conv_c3 = nn.Sequential(
            nn.Conv2d(channel, channel, int(self.kernel_size/kernel1/kernel2), int(self.kernel_size/kernel1/kernel2), 0,
                      groups=channel),
            nn.SiLU()
        )

    def forward(self, x):
        b, c, h, w = x[0].size()
        w_vi = self.avg_pool(x[0]).view(b, c)
        w_ir = self.avg_pool(x[1]).view(b, c)
        w_vi = self.fc(w_vi).view(b, c, 1, 1)
        w_ir = self.fc(w_ir).view(b, c, 1, 1)

        glob_t = self.compress(torch.cat([x[0], x[1]], 1))
        glob = self.conv_c3(self.conv_c2(self.conv_c1(glob_t))) if min(h, w) >= self.kernel_size else torch.mean(
                                                                                    glob_t, dim=[2, 3], keepdim=True)
        # Resize glob to match input spatial size (pyramid convs shrink without padding)
        if glob.shape[2:] != x[0].shape[2:]:
            glob = F.interpolate(glob, size=(h, w), mode='bilinear', align_corners=False)
        result_vi = x[0] * (self.act(w_ir * glob)).expand_as(x[0])
        result_ir = x[1] * (self.act(w_vi * glob)).expand_as(x[1])

        return result_vi, result_ir


class DEPA(nn.Module):
    """x0 --> RGB feature map,  x1 --> IR feature map"""
    def __init__(self, channel=512, m_kernel=None):
        super().__init__()
        self.conv1 = Conv(2, 1, 5)
        self.conv2 = Conv(2, 1, 5)
        self.compress1 = Conv(channel, 1, 3)
        self.compress2 = Conv(channel, 1, 3)
        self.act = nn.Sigmoid()

        """convolution merge"""
        if m_kernel is None:
            m_kernel = [3, 7]
        self.cv_v1 = Conv(channel, 1, m_kernel[0])
        self.cv_v2 = Conv(channel, 1, m_kernel[1])
        self.cv_i1 = Conv(channel, 1, m_kernel[0])
        self.cv_i2 = Conv(channel, 1, m_kernel[1])

    def forward(self, x):
        w_vi = self.conv1(torch.cat([self.cv_v1(x[0]), self.cv_v2(x[0])], 1))
        w_ir = self.conv2(torch.cat([self.cv_i1(x[1]), self.cv_i2(x[1])], 1))
        glob = self.act(self.compress1(x[0]) + self.compress2(x[1]))
        w_vi = self.act(glob + w_vi)
        w_ir = self.act(glob + w_ir)
        result_vi = x[0] * w_ir.expand_as(x[0])
        result_ir = x[1] * w_vi.expand_as(x[1])

        return result_vi, result_ir


class DEA3(nn.Module):
    """Three-modal fusion: x[0]=RGB, x[1]=IR, x[2]=Depth.

    Uses cross-gating with Depth as geometric guide, channel attention over all 3 modalities,
    and pyramid convolution for global context.
    """

    def __init__(self, channel=512, kernel_size=80, p_kernel=None, reduction=16):
        super().__init__()
        self.kernel_size = kernel_size
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.act = nn.Sigmoid()

        # Channel attention over 3 modalities
        self.fc = nn.Sequential(
            nn.Linear(channel * 3, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel * 3, bias=False),
            nn.Sigmoid()
        )

        # Pyramid convolution for global context
        if p_kernel is None:
            p_kernel = [5, 4]
        k1, k2 = p_kernel
        self.conv_c1 = nn.Sequential(nn.Conv2d(channel, channel, k1, k1, 0, groups=channel), nn.SiLU())
        self.conv_c2 = nn.Sequential(nn.Conv2d(channel, channel, k2, k2, 0, groups=channel), nn.SiLU())
        self.conv_c3 = nn.Sequential(
            nn.Conv2d(channel, channel, int(kernel_size / k1 / k2), int(kernel_size / k1 / k2), 0, groups=channel),
            nn.SiLU()
        )

        # Depth guides spatial attention for RGB and IR
        self.depth_guide = nn.Sequential(
            Conv(channel, channel // 2, 3),
            nn.Conv2d(channel // 2, 2, 1),
            nn.Sigmoid()
        )

        # Per-modal spatial weights (like DEPA)
        self.cv_r1 = Conv(channel, 1, 3)
        self.cv_r2 = Conv(channel, 1, 7)
        self.cv_i1 = Conv(channel, 1, 3)
        self.cv_i2 = Conv(channel, 1, 7)
        self.cv_d1 = Conv(channel, 1, 3)
        self.cv_d2 = Conv(channel, 1, 7)

        self.conv_r = Conv(2, 1, 5)
        self.conv_i = Conv(2, 1, 5)
        self.conv_d = Conv(2, 1, 5)

        # Global context compression (1x1 — no spatial mixing needed)
        self.compress_global = Conv(channel * 3, channel, 1)
        # Final fusion compression (3x3 — spatial refinement)
        self.compress_fuse = Conv(channel * 3, channel, 3)

    def forward(self, x):
        b, c, h, w = x[0].size()
        rgb, ir, dp = x[0], x[1], x[2]

        # 1. Channel attention from 3-modal global statistics
        pool_cat = torch.cat([self.avg_pool(xi).view(b, c) for xi in x], dim=1)  # (b, 3c)
        weights = self.fc(pool_cat).view(b, 3 * c, 1, 1)
        w_rgb, w_ir, w_dp = weights.chunk(3, dim=1)

        # 2. Global pyramid context (shared over all modalities via concat+compress_global)
        glob_t = self.compress_global(torch.cat([rgb, ir, dp], 1))
        glob = self.conv_c3(self.conv_c2(self.conv_c1(glob_t))) if min(h, w) >= self.kernel_size else \
            torch.mean(glob_t, dim=[2, 3], keepdim=True)
        if glob.shape[2:] != (h, w):
            glob = F.interpolate(glob, size=(h, w), mode='bilinear', align_corners=False)

        # 3. Depth-guided spatial attention (有界门控: [0.5, 1.0], depth 只能衰减50%, 永不摧毁特征)
        depth_guide = self.depth_guide(dp)  # (b, 2, h, w), sigmoid 输出 [0,1]
        g_rgb = 0.5 + 0.5 * depth_guide[:, 0:1, :, :]  # [0.5, 1.0]
        g_ir = 0.5 + 0.5 * depth_guide[:, 1:2, :, :]   # [0.5, 1.0]

        # 4. Per-modal spatial weights (DEPA-style)
        w_r = self.act(self.conv_r(torch.cat([self.cv_r1(rgb), self.cv_r2(rgb)], 1)))
        w_i = self.act(self.conv_i(torch.cat([self.cv_i1(ir), self.cv_i2(ir)], 1)))
        w_d = self.act(self.conv_d(torch.cat([self.cv_d1(dp), self.cv_d2(dp)], 1)))

        # 5. Cross-gating: each modality enhanced by others
        out_rgb = rgb * (w_rgb * glob).expand_as(rgb) * g_rgb  # Depth guides RGB
        out_ir = ir * (w_ir * glob).expand_as(ir) * g_ir       # Depth guides IR
        out_dp = dp * (w_dp * glob).expand_as(dp)

        # 6. Fuse and compress (3x3 for spatial refinement)
        fused = self.compress_fuse(torch.cat([out_rgb, out_ir, out_dp], 1))
        return self.act(fused)
