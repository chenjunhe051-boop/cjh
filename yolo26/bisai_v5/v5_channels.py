# -*- coding: utf-8 -*-
"""v5_channels -- 9 通道数据管线适配（重写 + 安装自证 + 独立自检）。

问题出在哪
----------
ultralytics 的 ``Format._format_img`` **只在通道数恰好为 3 时**做 BGR->RGB 翻转::

    img = img.transpose(2, 0, 1)
    img = np.ascontiguousarray(img[::-1] if random.uniform(0, 1) > self.bgr
                               and img.shape[0] == 3 else img)

9 通道输入因此保持 BGR，而 3 通道那条腿是 RGB —— 融合模型看到的 R/B 是反的。
旧实现是"整个替换 _format_img"，新版 ultralytics 只要实现细节变一点，替换就
静默失效（它的 print 是无条件的，不能当生效证据）。实测 identity delta：
yolo26m −0.232 / yolo11m −0.369。

本模块做三件事
--------------
1. ``install()``：**包装**（而不是替换）两个必要的适配，装完立刻**自证**，
   自证失败直接抛异常，不再允许静默失效：
     * BGR->RGB 只翻转前 3 个通道
     * ``cv2.copyMakeBorder`` 对 >4 通道改走 ``np.pad``（OpenCV 不支持 >4 通道）
2. ``selftest()``：同一批图分别走 3 通道 / 9 通道 val 管线，比较 ch0..2 是否
   逐比特一致；不一致时给出**差异类型诊断**（通道顺序反了 / 尺寸不同 / 数值偏移）
3. 把 ``v4_common`` 里的 ``install_padding_patch`` / ``install_format_patch``
   指向稳健版，避免任何调用点再踩旧坑

用法
----
    python bisai_v5/v5_channels.py --selftest --data /root/autodl-tmp/data_full --imgsz 1024
    python bisai_v5/v5_channels.py --install-check
"""
from __future__ import annotations

import argparse
import inspect
import math
import os
import random
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))            # upload_seed/bisai_v4

NAMES = ["person", "boat", "animal", "seat", "sign", "bicycle",
         "car", "ball", "light", "garbage_can", "uav", "tricycle"]
N_CH = 9

_APPLIED = {"fmt": False, "pad": False}
_PAD_TAIL = 0        # 通道 3..8 的 letterbox 填充值（训练/验证一致即可）


# --------------------------------------------------------------- patches
def _patch_format():
    """包装 Format._format_img：>3 通道时只把前 3 个通道翻成 RGB。"""
    import ultralytics.data.augment as A
    cur = A.Format._format_img
    if getattr(cur, "_v5_wrapped", False):
        _APPLIED["fmt"] = True
        return

    def _fmt(self, img):
        if img.ndim == 3 and img.shape[2] > 3 and random.uniform(0, 1) > self.bgr:
            img = img.copy()
            img[..., :3] = img[..., :3][..., ::-1]
        return cur(self, img)

    _fmt._v5_wrapped = True
    A.Format._format_img = _fmt
    _APPLIED["fmt"] = True


def _patch_pad(pad_tail=_PAD_TAIL):
    """cv2.copyMakeBorder 不支持 >4 通道；前 3 通道按原值，其余按 pad_tail 填。"""
    import cv2
    orig = cv2.copyMakeBorder
    if getattr(orig, "_v5_wrapped", False):
        _APPLIED["pad"] = True
        return

    def _pb(src, top, bottom, left, right, borderType, value=None):
        if src.ndim == 3 and src.shape[2] > 4:
            if borderType == cv2.BORDER_CONSTANT:
                head = value[0] if isinstance(value, (list, tuple)) else (value or 0)
                pad = [(top, bottom), (left, right), (0, 0)]
                out = [np.pad(src[..., :3], pad, mode="constant", constant_values=head)]
                if src.shape[2] > 3:
                    out.append(np.pad(src[..., 3:], pad, mode="constant",
                                      constant_values=pad_tail))
                return np.concatenate(out, axis=2)
            if borderType == cv2.BORDER_REPLICATE:
                return np.pad(src, [(top, bottom), (left, right), (0, 0)], mode="edge")
        # 注意：cv2 的第 7 个位置参数是 dst，不是 value —— 必须用关键字传，
        # 否则填充值会被丢掉、退化成 0（黑边）。v4_common 的旧版就踩了这个坑。
        return orig(src, top, bottom, left, right, borderType, value=value)

    _pb._v5_wrapped = True
    cv2.copyMakeBorder = _pb
    _APPLIED["pad"] = True


def _patch_grayscale():
    """给 3 通道训练管线补 grayscale 增强（你的 ultralytics 版本没有该参数）。

    包装 RandomHSV.forward：处理完颜色抖动后，以 V5_GRAYSCALE（默认0.15）概率
    把 RGB 图变灰。只对 3 通道生效，因此天然只影响 RGB 种子训练阶段；
    验证/测试（augment=False）不经过它，行为不变。
    """
    import cv2
    import ultralytics.data.augment as A
    cls = A.RandomHSV
    p = float(os.environ.get("V5_GRAYSCALE", "0.15"))

    def _wrap(cur):
        def _fwd(self, labels):
            labels = cur(self, labels)
            img = labels.get("img")
            if (p > 0 and random.random() < p and img is not None
                    and getattr(img, "ndim", 0) == 3 and img.shape[2] == 3):
                g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                labels["img"] = cv2.merge([g, g, g])
            return labels
        _fwd._v5_gray = True
        return _fwd

    # 不同版本入口名不同：有的叫 forward，有的只有 __call__
    for attr in ("forward", "__call__"):
        cur = getattr(cls, attr, None)
        if cur is not None and not getattr(cur, "_v5_gray", False):
            setattr(cls, attr, _wrap(cur))
            return


def _patch_base_loader():
    """中和 base.py 里被魔改过的 load_image（如果存在）。

    有些 YOLO-Master 代码库在 BaseDataset.load_image 里写死了"按 vis_->ir_/
    depth_ 拼 9 通道"，导致任何数据集（包括想要纯 RGB 的对照腿）都输出 9 通道。
    这里检测到这个魔改后，用一份干净的"仅 RGB"实现替换掉 —— 只影响本进程，
    不改动磁盘上的源码文件。未被魔改的环境（如朋友的机器）会自动跳过。
    """
    import cv2
    from ultralytics.data.base import BaseDataset
    try:
        src = inspect.getsource(BaseDataset.load_image)
    except Exception:                                       # noqa: BLE001
        return
    if getattr(BaseDataset.load_image, "_v5_rgb_override", False) or "vis_" not in src:
        return                                              # 未被魔改，无需处理
    print("[v5_channels] 检测到 base.py 的 load_image 被魔改成强制9通道，已临时中和（仅本进程）")

    def _load_image(self, i, rect_mode=True, resize_short=False):
        im, f, fn = self.ims[i], self.im_files[i], self.npy_files[i]
        if im is None:                                      # not cached in RAM
            if fn.exists():                                 # load npy
                try:
                    im = np.load(fn)
                    if im.ndim == 3 and im.shape[-1] > 3:   # 缓存的是魔改9通道，丢弃重读
                        Path(str(fn)).unlink(missing_ok=True)
                        im = cv2.imread(f)
                except Exception:                           # noqa: BLE001
                    try:
                        Path(str(fn)).unlink(missing_ok=True)
                    except Exception:                       # noqa: BLE001
                        pass
                    im = cv2.imread(f)                      # BGR
            else:                                           # read image
                im = cv2.imread(f)                          # BGR
            if im is None:
                raise FileNotFoundError("Image Not Found %s" % f)
            h0, w0 = im.shape[:2]
            imsz = self.imgsz
            if isinstance(imsz, (list, tuple)):
                imsz = max(imsz)
            imsz = int(imsz)
            if rect_mode:                                   # long side -> imgsz
                r = imsz / max(h0, w0)
                if r != 1:
                    w = min(math.ceil(w0 * r), imsz)
                    h = min(math.ceil(h0 * r), imsz)
                    im = cv2.resize(im, (w, h), interpolation=cv2.INTER_LINEAR)
            elif resize_short:                              # short side -> imgsz
                r = imsz / min(h0, w0)
                if r != 1:
                    w = min(math.ceil(w0 * r), imsz)
                    h = min(math.ceil(h0 * r), imsz)
                    im = cv2.resize(im, (w, h), interpolation=cv2.INTER_LINEAR)
            elif not (h0 == w0 == imsz):                    # square resize
                im = cv2.resize(im, (imsz, imsz), interpolation=cv2.INTER_LINEAR)
            # 兼容该 fork 的 Mosaic：它从 dataset.buffer 里抽样，原版 load_image
            # 负责填充；干净实现必须补上，否则 buffer 为空 ->
            # random.choices([], k=3) -> IndexError
            buf = getattr(self, "buffer", None)
            if isinstance(buf, list):
                buf.append(i)
            return im, (h0, w0), im.shape[:2]
        return self.ims[i], self.im_hw0[i], self.im_hw[i]

    _load_image._v5_rgb_override = True
    BaseDataset.load_image = _load_image


def verify_patches():
    """安装后自证：构造一个 9 通道 BGR 图，走一遍 Format，检查前 3 通道是否翻成 RGB。"""
    try:
        from ultralytics.data.augment import Format
        f = Format(bgr=0.0)
        img = np.zeros((4, 4, 9), dtype=np.uint8)
        img[..., 0] = 10      # B
        img[..., 1] = 20      # G
        img[..., 2] = 30      # R
        t = f._format_img(img.copy())
        if tuple(t.shape) != (9, 4, 4):
            return False, "Format 输出形状异常 %s" % (tuple(t.shape),)
        head = t[:3, 0, 0].tolist()
        if head != [30.0, 20.0, 10.0]:
            return False, ("前 3 通道没有翻成 RGB（得到 %s，期望 [30,20,10]）"
                           % head)
        import cv2
        src = np.zeros((2, 2, 9), dtype=np.uint8)
        out = cv2.copyMakeBorder(src, 1, 1, 1, 1, cv2.BORDER_CONSTANT,
                                 value=(114, 114, 114))
        if tuple(out.shape) != (4, 4, 9):
            return False, "copyMakeBorder 9 通道 padding 形状异常 %s" % (tuple(out.shape),)
        return True, "前 3 通道 BGR->RGB 正常；9 通道 padding 正常"
    except Exception as e:                     # noqa: BLE001
        return False, "自证过程异常: %r" % (e,)


def install(pad_tail=_PAD_TAIL, verbose=True):
    _patch_format()
    _patch_pad(pad_tail)
    _patch_base_loader()
    _patch_grayscale()
    ok, msg = verify_patches()
    if not ok:
        raise RuntimeError("[v5_channels] 安装自证失败 —— %s" % msg)
    # 旧入口指向稳健版，避免任何调用点再打一次（会双重翻转）
    try:
        import v4_common
        v4_common.install_format_patch = _already
        v4_common.install_padding_patch = _already
    except Exception:
        pass
    if verbose:
        print("[v5_channels] 9 通道适配已安装并通过自证：%s" % msg)
    return True


def _already(*a, **k):
    return None


# ------------------------------------------------------- 定位用的输入日志
_LOG_TAG = [""]
_IN_VAL = [False]        # 真实 val 循环开始后才置 True，用来跳过内部 warmup/AMP 自检


def _log_targets():
    """所有"实际会被调用"的 validator 类 —— DetectionValidator 自己重写了 preprocess。"""
    import ultralytics.engine.validator as V
    out = [("BaseValidator", V.BaseValidator)]
    try:
        from ultralytics.models.yolo.detect.val import DetectionValidator as DV
        if "preprocess" in DV.__dict__:
            out.append(("DetectionValidator", DV))
    except Exception:                                           # noqa: BLE001
        pass
    return out


def install_input_logger(tag="", n_batches=3):
    """打印"进入模型的第一个 batch"的通道统计。

    用途：identity 不过时，用它区分是【数据路径】还是【模型路径】出的问题 ——
    两条腿在同一 val 上、同 batch 顺序，所以 ch0..2 的均值/标准差必须一致。

    注意：只记录 **eval 模式的真实 val 批次** —— AMP 自检走的是 train 模式、
    输入是 640x640 的零张量，记下来没有意义（之前就踩过这个坑）。
    """
    _LOG_TAG[0] = tag                     # 每次调用刷新 tag，两条腿各打各的
    for name, cls in _log_targets():
        orig = cls.preprocess
        if getattr(orig, "_v5_logged", False):
            continue

        def _pp(self, batch, _orig=orig, _name=name):
            # 注意：preprocess 只会被**真实 val 批次**调用；而且此时 self.model 还是 None
            # （模型是 __call__ 里的局部变量），所以不能用 model.training 做判断。
            im = batch.get("img") if isinstance(batch, dict) else None
            if im is not None:
                _IN_VAL[0] = True
                k = getattr(self, "_v5_logged_batches", 0)
                if k < n_batches:
                    self._v5_logged_batches = k + 1
                    tag = _LOG_TAG[0]
                    try:
                        x = im.float() / 255.0
                        print("[v5-log %s] batch#%d 形状 %s (来自 %s)"
                              % (tag, k, tuple(im.shape), _name))
                        print("[v5-log %s]   均值 %s" % (
                            tag, [round(v, 5) for v in x.mean(dim=(0, 2, 3)).tolist()]))
                        print("[v5-log %s]   标准差 %s" % (
                            tag, [round(v, 5) for v in x.std(dim=(0, 2, 3)).tolist()]))
                        print("[v5-log %s]   校验和 sum=%.4f" % (tag, float(x.sum())))
                        print("[v5-log %s]   像素[0,0]=%s  像素[-1,-1]=%s" % (
                            tag, [round(v, 4) for v in x[0, :3, 0, 0].tolist()],
                            [round(v, 4) for v in x[0, :3, -1, -1].tolist()]))
                    except Exception as e:                          # noqa: BLE001
                        print("[v5-log %s] 统计失败: %r" % (tag, e))
            return _orig(self, batch)

        _pp._v5_logged = True
        cls.preprocess = _pp


def log_stem(model, tag="", n_ch=3):
    """给主干第一个 Conv 挂 hook：打印它实际收到的张量（形状 + 前 n_ch 通道均值）。"""
    try:
        layer = model.model[0]
    except Exception:                                               # noqa: BLE001
        print("[v5-log %s] 取不到 model[0]，跳过 stem 日志" % tag)
        return

    def hook(mod, inp, out):
        if (getattr(mod, "_v5_hooked", False) or getattr(mod, "training", False)
                or not _IN_VAL[0]):
            return
        mod._v5_hooked = True
        try:
            x = inp[0]
            if min(x.shape[-1], x.shape[-2]) <= 64:      # 跳过内部 32x32 warmup
                return
            m = [round(v, 4) for v in x.float().mean(dim=(0, 2, 3)).tolist()[:n_ch]]
            print("[v5-log %s] stem 输入 %s  前%d通道均值 %s"
                  % (tag, tuple(x.shape), n_ch, m))
            print("[v5-log %s] stem 输出 %s  均值 %.4f"
                  % (tag, tuple(out.shape), float(out.float().mean())))
        except Exception as e:                                      # noqa: BLE001
            print("[v5-log %s] stem 统计失败: %r" % (tag, e))

    layer.register_forward_hook(hook)


# ------------------------------------------------------------ 融合 identity 探针
def forwardtest(model_pt, imgsz=320, verbose=True):
    """张量级 identity 检验：同一次前向里比较"纯 RGB 模型"与"融合模型(warmup=0)"。

    为什么需要它：v5_initeval 要跑完整 val（几分钟）才能看出 identity 破了，
    而且看不出**破在哪一层**。这个探针只做一次前向，并顺带打印融合模块的
    权重范数与 warmup/gate 值，用来判断"零初始化"到底有没有生效。

    返回 0 表示通过，1 表示不通过。
    """
    import torch
    install(verbose=verbose)
    from ultralytics import YOLO
    from v5_model import apply_fusion_v5, set_inject_warmup, stem_in_channels

    ref = YOLO(str(model_pt))
    fus = YOLO(str(model_pt))
    apply_fusion_v5(fus.model)
    try:
        set_inject_warmup(fus.model, 0.0)
        print("[forwardtest] 已显式把 inject warmup 置 0")
    except Exception as e:                                  # noqa: BLE001
        print("[forwardtest] set_inject_warmup 调用失败: %r" % e)

    print("[forwardtest] 融合模块权重范数（零初始化的那几层应该 ≈0）:")
    for name, mod in fus.model.named_modules():
        if not any(k in name for k in ("inject", "aux", "fuse", "reliab", "gate")):
            continue
        s, n = 0.0, 0
        for p in mod.parameters(recurse=False):
            s += float(p.detach().float().abs().sum())
            n += p.numel()
        if n:
            print("   %-40s |W|1=%10.6f  (参数 %d)" % (name, s, n))

    torch.manual_seed(0)
    # 鉴别诊断：该 fork 的 _predict_once 可能有私货（多任务/深度头等），
    # 若 ref 与"朴素逐层循环"不一致，说明融合前向必须复刻那段私货才严格 identity
    from v5_model import _rgb_forward_only
    ref_ch = stem_in_channels(ref.model)
    if ref_ch == N_CH:
        # legacy 模式：参考模型直接吃老格式 9 通道；融合模型吃 v5 格式，
        # 但构造 v5 输入使其经 legacy 映射后与参考输入逐值一致
        print("[forwardtest] 检测到 9 通道 stem，启用 legacy 对照")
        x_ref = torch.rand(1, 9, imgsz, imgsz)
        x9 = torch.cat([x_ref[:, :3][:, [2, 1, 0]], x_ref[:, 3:4], x_ref[:, 6:7],
                        torch.rand(1, 4, imgsz, imgsz)], dim=1)
    else:
        x_ref = torch.rand(1, 3, imgsz, imgsz)
        x9 = torch.cat([x_ref, torch.rand(1, 6, imgsz, imgsz)], dim=1)
    ref.model.eval()
    fus.model.eval()
    with torch.no_grad():
        y_rgb = ref.model(x_ref)
        y_plain = _rgb_forward_only(ref.model, x_ref)
        y_fus = fus.model(x9)
    a = y_rgb[0] if isinstance(y_rgb, (tuple, list)) else y_rgb
    b = y_fus[0] if isinstance(y_fus, (tuple, list)) else y_fus
    d_ref = float((torch.as_tensor(y_rgb).float().flatten() -
                   torch.as_tensor(y_plain).float().flatten()).abs().max())
    print("[forwardtest] ref vs 朴素逐层循环 最大差 = %.6e  (%s)"
          % (d_ref, "fork前向是标准的" if d_ref <= 1e-5 else "fork前向有私货!"))
    print("[forwardtest] 输出形状: rgb %s | fusion %s" % (tuple(a.shape), tuple(b.shape)))
    if tuple(a.shape) != tuple(b.shape):
        print("[forwardtest] FAIL —— 输出形状不同，说明前向派发有问题")
        return 1
    d = float((a - b).abs().max())
    print("[forwardtest] 最大绝对差 = %.6e" % d)
    if d <= 1e-5:
        print("[forwardtest] PASS —— warmup=0 时融合模型逐值等于 RGB 模型（地板成立）")
        return 0
    print("[forwardtest] FAIL —— 融合模块的随机权重泄漏到输出了：")
    print("             说明这些层没有真正零初始化，或 warmup 没被采纳。")
    print("             上面权重范数里，凡是本应为 0 却明显非 0 的那几行就是元凶。")
    return 1


# --------------------------------------------------------------- selftest
def _build(data, imgsz, channels):
    from ultralytics.data.dataset import YOLODataset
    data = Path(data)
    val = data / "splits" / "val.txt"
    spec = {"names": {i: n for i, n in enumerate(NAMES)}, "channels": channels}
    if channels == 3:
        return YOLODataset(img_path=str(val), data=spec, task="detect", augment=False,
                           imgsz=imgsz, rect=True, stride=32, pad=0.0, cache=False,
                           single_cls=False)
    from v5_dataset import TriModalDatasetV4
    TriModalDatasetV4.TRAIN_IR = data / "ir"
    TriModalDatasetV4.VAL_IR = data / "ir"
    TriModalDatasetV4.TRAIN_DEPTH = data / "depth"
    TriModalDatasetV4.VAL_DEPTH = data / "depth"
    return TriModalDatasetV4(img_path=str(val), data=spec, task="detect", augment=False,
                             imgsz=imgsz, rect=True, stride=32, pad=0.0, cache=False,
                             single_cls=False)


def selftest(data, imgsz=1024, n=3, verbose=True):
    """3 通道 vs 9 通道 val 管线，逐比特比较 ch0..2。"""
    install(verbose=verbose)
    ds3, ds9 = _build(data, imgsz, 3), _build(data, imgsz, 9)
    print("[selftest] 3ch 数据集 %d 张 / 9ch 数据集 %d 张" % (len(ds3), len(ds9)))
    idx9 = {Path(p).name: i for i, p in enumerate(ds9.im_files)}
    checked = bad = 0
    for i in range(len(ds3)):
        name = Path(ds3.im_files[i]).name
        if name not in idx9:
            continue
        a = ds3[i]["img"]            # (3,H,W) float tensor
        b = ds9[idx9[name]]["img"]   # (9,H,W)
        checked += 1
        a3 = a.numpy() if hasattr(a, "numpy") else np.asarray(a)
        b9 = b.numpy() if hasattr(b, "numpy") else np.asarray(b)
        if a3.shape != b9[:3].shape:
            print("  [FAIL] %s 形状不同: 3ch %s vs 9ch[:3] %s"
                  % (name, a3.shape, b9[:3].shape))
            bad += 1
        else:
            d = float(np.abs(a3 - b9[:3]).max())
            if d <= 1e-6:
                print("  [OK]   %s ch0..2 逐比特一致" % name)
            else:
                rev = float(np.abs(a3 - b9[:3][::-1]).max())
                why = "通道顺序反了（patch 没生效）" if rev <= 1e-6 else \
                      "数值/填充不一致"
                print("  [FAIL] %s ch0..2 最大差 %.4f → %s" % (name, d, why))
                bad += 1
        if checked >= n:
            break
    print("[selftest] 检查 %d 张，失败 %d 张" % (checked, bad))
    if bad == 0 and checked:
        print("[selftest] PASS —— 9 通道管线的 ch0..2 与 3 通道一致，identity 前提成立")
        return 0
    print("[selftest] FAIL —— 先修这里，再谈融合训练")
    return 1


def main():
    ap = argparse.ArgumentParser(description="9 通道数据管线适配与自检")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--install-check", action="store_true")
    ap.add_argument("--forwardtest", action="store_true",
                    help="张量级 identity 探针（需 --model），判断零初始化/warmup 是否生效")
    ap.add_argument("--model", default=None)
    ap.add_argument("--data", default="/root/autodl-tmp/data_full")
    ap.add_argument("--imgsz", type=int, default=1024)
    ap.add_argument("--n", type=int, default=3)
    args = ap.parse_args()
    if args.forwardtest:
        if not args.model:
            sys.exit("[error] --forwardtest 需要 --model <checkpoint>")
        return forwardtest(args.model, imgsz=args.imgsz)
    if args.selftest:
        return selftest(args.data, args.imgsz, args.n)
    install()
    print("[v5_channels] install-check OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
