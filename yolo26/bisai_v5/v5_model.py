# -*- coding: utf-8 -*-
"""v5 tri-modal fusion for YOLO11 (ultralytics) -- "TABFuse".

Reference mapping (the 5 supplied papers)
-----------------------------------------


Channel contract (identical for train / val / inference).  N_CH = 9
    0..2  RGB            (exactly what cv2.imread returns; ultralytics only
                          flips BGR->RGB when the channel count is exactly 3,
                          so the layout is stable end to end)
    3     IR             uint8 0..255
    4     depth linear   0..255
    5     depth inverse  0..255  (near = bright)
    6     depth valid    0 or 255
    7     thermal saliency = (IR - luma(RGB)) / 2 + 128   (new)
    8     depth edge energy = |Sobel(depth inverse)|        (new)

Safety (carried over from v4.2 "RGB-lock", this is what protects the score)
    * the RGB stream is the untouched pretrained detector;
    * every fused path is zero-initialised AND ramped by SwitchInject.warmup,
      so warmup == 0 reproduces the RGB model bit-identically
      -> best.pt can never fall below the RGB seed.
"""
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

N_CH = 9
FUSION_LAYERS = [int(x) for x in os.environ.get("V5_FUSION_LAYERS", "4,6,10").split(",")]
# 前向路径开关：训练时由 apply_fusion_v5 从 env 固化到模型属性并镜像到这里；
# SwitchInject 读这个表——保证训练/评估/提交三端行为一致，不再依赖 env。
_PATH_FLAGS = {"additive": os.environ.get("V5_ADDITIVE", "0") == "1",
               "xattn": os.environ.get("V5_XATTN", "0") == "1"}
INJECT_BIAS = float(os.environ.get("V5_INJECT_BIAS", "2.0"))
AUX_ATTRS = ("aux_encoder", "inject", "reliability")

IDX_IR = 3
IDX_SAL = 7
IDX_GEOM = (4, 5, 6, 8)


def _zero_init_conv(conv):
    """Zero the conv weights (not the BN gamma) so the stream starts at 0.

    Zeroing BN gamma would kill the gradient of the whole branch; zeroing the
    conv keeps the forward output at 0 while still letting the branch learn.
    """
    if getattr(conv, "bias", None) is not None:
        nn.init.zeros_(conv.bias)
    nn.init.zeros_(conv.weight)


class ConvBNAct(nn.Module):
    def __init__(self, c_in, c_out, k=3, s=1, p=None):
        super().__init__()
        p = k // 2 if p is None else p
        self.conv = nn.Conv2d(c_in, c_out, k, s, p, bias=False)
        self.bn = nn.BatchNorm2d(c_out)
        self.act = nn.SiLU()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class ECA(nn.Module):
    """Efficient channel attention -- the switch signal of paper [1]."""

    def __init__(self, channels, k_size=5):
        super().__init__()
        self.avg = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=k_size // 2, bias=False)

    def forward(self, x):
        y = self.avg(x).flatten(1)                  # (B, C)
        y = self.conv(y.unsqueeze(1)).squeeze(1)    # (B, C)
        return torch.sigmoid(y)[..., None, None]    # (B, C, 1, 1)


class OrthoProj(nn.Module):
    """Per-pixel Gram-Schmidt decorrelation of the thermal / geometry streams.

    ft <- ft - a * <ft, fg> / ||fg||^2 * fg
    fg <- fg - b * <ft, fg> / ||ft||^2 * ft

    a and b are learnable so the network can decide how much redundancy to
    remove (paper [5] uses exactly this to stop thermal and depth cues from
    being counted twice).
    """

    def __init__(self):
        super().__init__()
        self.alpha = nn.Parameter(torch.tensor(1.0))
        self.beta = nn.Parameter(torch.tensor(1.0))

    def forward(self, ft, fg):
        eps = 1e-5
        dot = (ft * fg).sum(1, keepdim=True)
        ft_out = ft - self.alpha * (dot / ((fg * fg).sum(1, keepdim=True) + eps)) * fg
        fg_out = fg - self.beta * (dot / ((ft * ft).sum(1, keepdim=True) + eps)) * ft
        return ft_out, fg_out


class StreamDown(nn.Module):
    """Stride-2 downsample of both modality streams (their channel counts differ)."""

    def __init__(self, c_t, c_g, width):
        super().__init__()
        self.t = ConvBNAct(c_t, width, 3, 2)
        self.g = ConvBNAct(c_g, width, 3, 2)

    def forward(self, t, g):
        return self.t(t), self.g(g)


class AuxLevel(nn.Module):
    """FMAB-inspired fusion level (paper [4]).

    Both modalities first go through ONE parameter-shared conv (the
    modality-common structure) and then through their own parameter-specific
    conv (thermal-only / geometry-only).  The two specific maps are
    decorrelated, the four maps are concatenated and re-weighted by a channel
    attention gate, and a 1x1 conv produces the level feature.
    """

    def __init__(self, c_t, c_g, width, out_c):
        super().__init__()
        self.down = StreamDown(c_t, c_g, width)
        self.shared = ConvBNAct(width, width, 3, 1)
        self.spec_t = ConvBNAct(width, width, 3, 1)
        self.spec_g = ConvBNAct(width, width, 3, 1)
        self.ortho = OrthoProj()
        self.attn = ECA(4 * width)
        self.fuse = ConvBNAct(4 * width, out_c, 1)
        _zero_init_conv(self.fuse.conv)

    def forward(self, t, g, gate_t, gate_g):
        t_d, g_d = self.down(t, g)
        s_t = self.shared(t_d)
        s_g = self.shared(g_d)
        u_t = self.spec_t(t_d) * gate_t
        u_g = self.spec_g(g_d) * gate_g
        u_t, u_g = self.ortho(u_t, u_g)
        cat = torch.cat([s_t, s_g, u_t, u_g], 1)
        return self.fuse(cat * self.attn(cat)), t_d, g_d


class BiFuse(nn.Module):
    """Bi-directional progressive aggregation of the aux levels (paper [4]).

    top-down  : deep (semantic) guidance is folded into the shallower levels;
    bottom-up : the refined shallow level is folded back into the deeper ones.
    All projections are zero-initialised, so the module starts as identity and
    can only ever ADD information.
    """

    def __init__(self, c3, c4, c5):
        super().__init__()
        self.td_5to4 = ConvBNAct(c5, c4, 1, 1)
        self.td_4to3 = ConvBNAct(c4, c3, 1, 1)
        self.bu_3to4 = ConvBNAct(c3, c4, 1, 1)
        self.bu_4to5 = ConvBNAct(c4, c5, 1, 1)
        for conv in (self.td_5to4.conv, self.td_4to3.conv, self.bu_3to4.conv, self.bu_4to5.conv):
            _zero_init_conv(conv)

    def forward(self, a3, a4, a5):
        # top-down: deep semantics are upsampled into the shallower levels
        a4_t = a4 + F.interpolate(self.td_5to4(a5), size=a4.shape[2:], mode="nearest")
        a3_t = a3 + F.interpolate(self.td_4to3(a4_t), size=a3.shape[2:], mode="nearest")
        # bottom-up: the refined shallow level is folded back into the deeper ones
        a4_b = a4_t + self.bu_3to4(F.interpolate(a3_t, size=a4_t.shape[2:], mode="nearest"))
        a5_b = a5 + self.bu_4to5(F.interpolate(a4_b, size=a5.shape[2:], mode="nearest"))
        return a3_t, a4_b, a5_b


class Reliability(nn.Module):
    """Per-modality reliability gates from global per-image statistics.

    This is the differentiable version of the question EvaNet (paper [3])
    asks when it scores a fused image: how much usable information did each
    modality actually contribute?  A dark RGB frame / a depth map that is
    mostly invalid / a flat thermal image pushes the corresponding gate down,
    which is exactly the competition's robustness requirement.
    """

    def __init__(self, hidden=16):
        super().__init__()
        self.thermal = self._mlp(5, hidden)
        self.geometry = self._mlp(4, hidden)

    @staticmethod
    def _mlp(k, hidden):
        m = nn.Sequential(nn.Linear(k, hidden), nn.SiLU(), nn.Linear(hidden, 1))
        with torch.no_grad():
            m[-1].bias.fill_(3.0)      # sigmoid(3) ~ 0.95 -> gates start open
        return m

    @staticmethod
    def stats(x):
        ir = x[:, IDX_IR]
        sal = x[:, IDX_SAL] - 0.5          # (IR - luma)/2 + 128 over 255 -> centred
        d_lin = x[:, 4]
        d_val = x[:, 6]
        d_edge = x[:, 8]
        t = torch.stack([
            ir.mean(dim=(1, 2)),
            ir.std(dim=(1, 2)),
            sal.abs().mean(dim=(1, 2)),
            sal.mean(dim=(1, 2)),
            (sal > 0.06).to(sal.dtype).mean(dim=(1, 2)),
        ], dim=1)
        g = torch.stack([
            d_val.mean(dim=(1, 2)),
            d_lin.mean(dim=(1, 2)),
            d_lin.std(dim=(1, 2)),
            d_edge.mean(dim=(1, 2)),
        ], dim=1)
        return t, g

    def forward(self, x):
        t, g = self.stats(x)
        gate_t = torch.sigmoid(self.thermal(t)).view(-1, 1, 1, 1)
        gate_g = torch.sigmoid(self.geometry(g)).view(-1, 1, 1, 1)
        if not self.training:
            self._dbg = dict(gate_t=float(gate_t.mean()), gate_g=float(gate_g.mean()))
        return gate_t, gate_g


class AuxEncoderV5(nn.Module):
    """Small 2-stream (thermal / geometry) encoder producing P3/P4/P5 aux maps."""

    def __init__(self, channels):
        super().__init__()
        c3, c4, c5 = channels
        self.stem_t = ConvBNAct(2, 24, 3, 2)     # /2
        self.stem_g = ConvBNAct(4, 24, 3, 2)     # /2
        self.pre = StreamDown(24, 24, 32)        # /4
        self.lvl3 = AuxLevel(32, 32, 48, c3)     # /8
        self.lvl4 = AuxLevel(48, 48, 64, c4)     # /16
        self.lvl5 = AuxLevel(64, 64, 96, c5)     # /32
        self.bifuse = BiFuse(c3, c4, c5)

    def forward(self, x, gate_t, gate_g):
        t = torch.cat([x[:, IDX_IR:IDX_IR + 1], x[:, IDX_SAL:IDX_SAL + 1]], 1)
        g = x[:, list(IDX_GEOM)]
        t = self.stem_t(t)
        g = self.stem_g(g)
        t, g = self.pre(t, g)
        a3, t, g = self.lvl3(t, g, gate_t, gate_g)
        a4, t, g = self.lvl4(t, g, gate_t, gate_g)
        a5, t, g = self.lvl5(t, g, gate_t, gate_g)
        return self.bifuse(a3, a4, a5)


class MageGate(nn.Module):
    """MAGE 式门控（arXiv:2604.16630）：通道门 + 空间门，只调制交叉残差，不动恒等路径。

    全局统计(avg+max) -> MLP -> 逐通道门 gc；a 的 mean/max 空间证据 -> 7x7 conv -> 逐像素门 gs。
    与 xattn 组合 = 论文 MAGE+BiTE 的轻量复刻。
    """

    def __init__(self, c):
        super().__init__()
        self.ch = nn.Sequential(nn.Linear(2 * c, max(8, c // 4)), nn.SiLU(), nn.Linear(max(8, c // 4), c))
        self.sp = nn.Sequential(nn.Conv2d(2, 1, 7, padding=3), nn.Sigmoid())
        # QA2FDet 启发：模态差异分支（WD）。Δ=rgb-a 的统计量携带"哪里不一致"
        # 的信号——消融中这是全场最大单项增益（+3.8%）。
        self.dh = nn.Sequential(nn.Linear(2 * c, max(8, c // 4)), nn.SiLU(), nn.Linear(max(8, c // 4), c))

    def forward(self, rgb, a):
        z = torch.cat([rgb, a], 1)
        gap, gmp = z.mean(dim=(2, 3)), z.amax(dim=(2, 3))
        gc = torch.sigmoid(self.ch(gap + gmp)).view(z.shape[0], -1, 1, 1)
        gs_in = torch.cat([a.mean(1, keepdim=True), a.amax(1, keepdim=True)], 1)
        d = rgb - a
        d_mu, d_sd = d.mean(dim=(2, 3)), d.std(dim=(2, 3))
        gd = torch.sigmoid(self.dh(torch.cat([d_mu, d_sd], 1))).view(z.shape[0], -1, 1, 1)
        return gc * self.sp(gs_in) * gd


class CrossAttn(nn.Module):
    """跨模态注意力（文献：CMAFF 一脉的 RGB-T 融合机制）。

    辅助特征作 Query 去查询 RGB 特征："哪里需要热/深度信息"。
    gamma 零初始化 => 起步严格恒等，地板锁不变。
    只在 P5（32x32 特征图）使用，计算量可控。
    """

    def __init__(self, c):
        super().__init__()
        self.q = nn.Conv2d(c, c, 1, bias=False)
        self.k = nn.Conv2d(c, c, 1, bias=False)
        self.v = nn.Conv2d(c, c, 1, bias=False)
        self.out = nn.Conv2d(c, c, 1, bias=False)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, rgb, a):
        b, c, h, w = rgb.shape
        hw = h * w
        if hw > 4096:          # 只在深层小特征图上做，防显存/算力爆炸
            return torch.zeros_like(rgb)
        q = self.q(a).flatten(2)                    # B,C,HW
        k = self.k(rgb).flatten(2)
        v = self.v(rgb).flatten(2)
        attn = torch.softmax(torch.bmm(q.transpose(1, 2), k) / (c ** 0.5), dim=-1)
        o = torch.bmm(v, attn.transpose(1, 2)).view(b, c, h, w)
        return self.gamma * self.out(o)


class SwitchInject(nn.Module):
    """CSSA-style injection of an aux feature into an RGB feature (paper [1]).

    Per channel the ECA weights of RGB and aux are compared; the soft switch
    ``sigmoid(gain * (w_aux - w_rgb) + bias)`` decides how much of the RGB
    channel is replaced by the aux channel.  A parameter-free spatial
    attention (mean + max pooling) then keeps only the locations that matter.
    ``warmup`` is a plain python attribute owned by the trainer: at 0.0 the
    module returns the RGB feature untouched (hard identity floor).
    """

    def __init__(self, c):
        super().__init__()
        self.proj = ConvBNAct(c, c, 1)
        self.eca_rgb = ECA(c)
        self.eca_aux = ECA(c)
        self.gain = nn.Parameter(torch.tensor(6.0))
        self.bias = nn.Parameter(torch.tensor(INJECT_BIAS))
        self.spatial = nn.Sequential(nn.Conv2d(2, 1, 7, padding=3), nn.Sigmoid())
        self.xattn = CrossAttn(c)
        self.mage = MageGate(c)
        self.warmup = 1.0

    def forward(self, rgb, aux, gate):
        if aux.shape[2:] != rgb.shape[2:]:
            aux = F.interpolate(aux, size=rgb.shape[2:], mode="bilinear", align_corners=False)
        a = self.proj(aux) * gate
        w_rgb = self.eca_rgb(rgb)
        w_aux = self.eca_aux(a)
        switch = torch.sigmoid(self.gain * (w_aux - w_rgb) + self.bias)
        if _PATH_FLAGS.get("additive", False):
            # 加法注入（逼学红外/深度版）：aux 以增量身份进入主干，不再有 switch 淘汰。
            # proj 零初始化 + warmup 升温 => warmup=0 时严格恒等，地板锁不变。
            attn = self.spatial(torch.cat([rgb.mean(1, keepdim=True),
                                           rgb.amax(1, keepdim=True)], 1))
            if _PATH_FLAGS.get("xattn", False):
                gate = self.mage(rgb, a)
                return rgb + self.warmup * (attn * a * gate + self.xattn(rgb, a))
            if not self.training:
                self._dbg = dict(switch=-1.0, attn=float(attn.mean()),
                                 a_abs=float(a.abs().mean()), rgb_abs=float(rgb.abs().mean()),
                                 warmup=float(self.warmup), mod=float(a.abs().mean()))
            return rgb + self.warmup * (attn * a)
        mixed = rgb + switch * (a - rgb)
        attn = self.spatial(torch.cat([mixed.mean(1, keepdim=True),
                                       mixed.amax(1, keepdim=True)], 1))
        if not self.training:
            self._dbg = dict(switch=float(switch.mean()), attn=float(attn.mean()),
                             a_abs=float(a.abs().mean()), rgb_abs=float(rgb.abs().mean()),
                             warmup=float(self.warmup), mod=float((mixed - rgb).abs().mean()))
        return rgb + self.warmup * (attn * (mixed - rgb))


def set_inject_warmup(det, value):
    """Set the identity-ramp factor (0..1) on every SwitchInject module."""
    value = max(0.0, min(1.0, float(value)))
    if hasattr(det, "inject"):
        for inj in det.inject:
            if hasattr(inj, "warmup"):
                inj.warmup = value
    return value


def stem_in_channels(model):
    """第一个卷积的输入通道数：3 = 纯RGB种子，9 = 老的三模态拼接模型。"""
    for m in model.model.modules():
        if isinstance(m, nn.Conv2d):
            return int(m.in_channels)
    return 3


def stem_input_9ch(x, legacy9):
    """把 v5 的 9 通道数据格式转成模型主干实际期望的输入。

    legacy9=False（朋友的RGB种子）：直接取前3通道。
    legacy9=True（你的 y26x_o365 等老模型）：v5格式 [RGB, IR灰, 深度lin, inv, valid, sal, edge]
        -> 老格式 [BGR, IR灰x3, 深度linx3]（RGB翻回BGR；IR/深度各复制3份，
        与老 base.py 的 dstack([rgb, ir_bgr, dp3]) 逐值一致，IR三通道相同的前提
        下红外图 B=G=R）。该映射是确定性的，因此 warmup=0 时融合模型与原模型
        逐值相等 —— 地板 = 你的48分模型本身。
    """
    if legacy9:
        return torch.cat([x[:, :3][:, [2, 1, 0]],
                          x[:, 3:4].expand(-1, 3, -1, -1),
                          x[:, 4:5].expand(-1, 3, -1, -1)], 1)
    return x[:, :3]


def _probe_channels(model):
    """Return the channel counts of FUSION_LAYERS from a dummy forward pass."""
    dev = next(model.parameters()).device
    ch = stem_in_channels(model)
    shapes = {}
    handles = []
    for idx in FUSION_LAYERS:
        handles.append(model.model[idx].register_forward_hook(
            lambda m, inp, out, i=idx: shapes.__setitem__(i, int(out.shape[1]))))
    was_train = model.training
    model.eval()
    with torch.no_grad():
        model(torch.zeros(1, ch, 64, 64, device=dev))
    if was_train:
        model.train()
    for h in handles:
        h.remove()
    return [shapes[i] for i in FUSION_LAYERS]


def _rgb_forward_only(self, x, profile=False):
    """The untouched ultralytics forward, used for every non-tri-modal call."""
    y = []
    for m in self.model:
        if m.f != -1:
            x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]
        x = m(x)
        y.append(x if m.i in self.save else None)
    return x


def _v5_predict_once_impl(self, x, profile=False, visualize=False, embed=None):
    """Tri-modal forward: 9-ch input, RGB stream + injected aux."""
    if (not getattr(self, "_trimodal", False)) or x is None or x.dim() != 4 or x.shape[1] != N_CH:
        return _rgb_forward_only(self, x, profile)

    # the reliability gates stay in the graph on purpose: they are trainable,
    # so the model can learn *when* the thermal / geometry stream is usable
    # instead of trusting a hand-set constant.
    gate_t, gate_g = self.reliability(x)
    gate = gate_t * gate_g
    aux = self.aux_encoder(x, gate_t, gate_g)

    y, dt = [], []
    x = stem_input_9ch(x, getattr(self, "_v5_legacy9", False))
    for m in self.model:
        if m.f != -1:
            x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]
        if profile:
            self._profile_one_layer(m, x, dt)
        x = m(x)
        if m.i in FUSION_LAYERS:
            x = self.inject[FUSION_LAYERS.index(m.i)](x, aux[FUSION_LAYERS.index(m.i)], gate)
        y.append(x if m.i in self.save else None)
    return x


_CLASS_PATCHED = [False]
_ORIG_PREDICT_ONCE = None


def _install_class_patch():
    """Install the class-level forward dispatcher once (idempotent)."""
    global _ORIG_PREDICT_ONCE
    if _CLASS_PATCHED[0]:
        return
    from ultralytics.nn.tasks import DetectionModel
    _ORIG_PREDICT_ONCE = DetectionModel._predict_once

    def _dispatch(self, *args, **kwargs):
        if getattr(self, "_trimodal", False):
            x = args[0] if args else kwargs.get("x")
            profile = bool(args[1]) if len(args) > 1 else bool(kwargs.get("profile", False))
            return _v5_predict_once_impl(self, x, profile=profile)
        return _ORIG_PREDICT_ONCE(self, *args, **kwargs)

    DetectionModel._predict_once = _dispatch
    _CLASS_PATCHED[0] = True
    print("[v5] class-level forward dispatcher installed")


def _load_aux_weights(det, state_dict):
    """Copy aux module weights from a checkpoint state dict (or nn.Module)."""
    if state_dict is None:
        return
    sd = state_dict.state_dict() if isinstance(state_dict, torch.nn.Module) else state_dict
    total = 0
    for attr in AUX_ATTRS:
        mod = getattr(det, attr, None)
        if mod is None:
            continue
        pref = attr + "."
        part = {k[len(pref):]: v for k, v in sd.items() if k.startswith(pref)}
        if part:
            res = mod.load_state_dict(part, strict=False)
            total += len(part)
            if res.missing_keys:
                print("[v5] aux %s: %d keys loaded, missing %d" % (attr, len(part), len(res.missing_keys)))
            else:
                print("[v5] aux %s: %d keys restored from checkpoint" % (attr, len(part)))
    print("[v5] aux weights restored (%d tensors)" % total if total else "[v5] no aux weights found")


def apply_fusion_v5(model, state_dict=None):
    """Attach the v5 fusion modules and install the class-level dispatcher.

    ``model`` is an ultralytics DetectionModel (or a YOLO wrapper).  Idempotent.
    ``state_dict`` restores aux weights after a ``YOLO(path)`` reload.
    """
    _install_class_patch()
    if hasattr(model, "model") and hasattr(model.model, "__getitem__"):
        det = model
    elif (hasattr(model, "model") and hasattr(model.model, "model")
          and hasattr(model.model.model, "__getitem__")):
        det = model.model
    else:
        print("[v5] not a DetectionModel, skip fusion")
        return

    if not hasattr(det, "aux_encoder"):
        channels = _probe_channels(det)
        print("[v5] fusion channels %s" % dict(zip(FUSION_LAYERS, channels)))
        dev = next(det.parameters()).device
        det.aux_encoder = AuxEncoderV5(channels).to(dev)
        det.inject = nn.ModuleList([SwitchInject(c) for c in channels]).to(dev)
        det.reliability = Reliability().to(dev)
        print("[v5] FMAB aux encoder + bi-fuse + switch-inject + reliability attached")
    det._trimodal = True
    det._v5_additive = bool(getattr(det, "_v5_additive", _PATH_FLAGS["additive"]))
    det._v5_xattn = bool(getattr(det, "_v5_xattn", _PATH_FLAGS["xattn"]))
    _PATH_FLAGS["additive"] = det._v5_additive
    _PATH_FLAGS["xattn"] = det._v5_xattn
    if state_dict is not None:
        # 从 checkpoint 恢复的是"已训练"融合模型：评估时注入应全开（warmup  ramp 仅训练期使用）
        set_inject_warmup(det, 1.0)
    det._v5_legacy9 = bool(getattr(det, "_v5_legacy9", False)) or (stem_in_channels(det) == N_CH)
    if det._v5_legacy9:
        print("[v5] 检测到9通道stem（老的三模态拼接模型）-> 启用legacy映射，地板=该模型本身")
    print("[v5] model flagged tri-modal (%d channels)" % N_CH)
    _load_aux_weights(det, state_dict)


# drop-in alias: existing v4 scripts can import apply_fusion_v4 from v5_model
apply_fusion_v4 = apply_fusion_v5