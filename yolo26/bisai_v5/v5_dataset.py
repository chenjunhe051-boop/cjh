# -*- coding: utf-8 -*-
"""v5 9-channel tri-modal dataset.

Channel contract (identical for train / val / inference):

    ch0..2  RGB            exactly what cv2.imread returns (BGR).  ultralytics
                           only flips BGR->RGB when the channel count is 3, so
                           a 9-channel tensor is passed through untouched.
    ch3     IR             uint8 0..255
    ch4     depth linear   0..255
    ch5     depth inverse  0..255  (near = bright)
    ch6     depth valid    0 or 255
    ch7     thermal saliency = (IR - luma(RGB)) / 2 + 128     <-- new in v5
    ch8     depth edge energy = |Sobel(depth inverse)|        <-- new in v5

Two behaviours the stock ultralytics pipeline cannot give us:

  * depth dtype branch.  149 of the 2000 depth files are 8-bit JPEGs
    (640x360, already normalised to 0..255), not uint16 millimetres.
    encode_depth_v5() detects that and passes them through, instead of
    scaling by 1/19999 which crushed those channels to black.

  * photometric jitter.  ultralytics' RandomHSV returns immediately unless
    the image has exactly 3 channels, so a 9-channel batch received NO colour
    augmentation at all (hsv_* hyper-parameters were silently ignored).
    photometric_jitter9() jitters ch0..3 (BGR + IR) and recomputes ch7 from
    the jittered pair, leaving all four depth channels bit-identical -- the
    same invariant aug_pipeline.py verifies as I3.

The two extra channels are *physical priors*: with only 2000 training images a
small auxiliary encoder cannot be expected to rediscover "hot but not bright"
(the signature of a person under a street lamp at night) or the depth
discontinuity that separates two touching objects.  Giving them explicitly is
cheap and is one of the main reasons v5 can make the IR stream useful at all.
"""
import math
import os
import random
from copy import deepcopy
from pathlib import Path

import cv2
import numpy as np
from ultralytics.data.dataset import YOLODataset

# ---- 自包含：原来从 v4_common 导入的三个小工具，这里内联实现 ----
IMG_EXTS = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')


def _find_modal_any(dirs, stem):
    """在多个候选目录里找 stem 的模态图（val 里可能混有来自 train 目录的图）。"""
    if isinstance(dirs, (str, Path)):
        dirs = [dirs]
    for d in dirs:
        p = _find_modal(d, stem)
        if p is not None:
            return p
    return None


def _find_modal(d, stem):
    """在目录 d 里找 stem 对应的模态图像（兼容大小写扩展名和常见前缀）。"""
    d = Path(d)
    for ext in IMG_EXTS:
        for name in (stem + ext, stem + ext.upper()):
            p = d / name
            if p.is_file():
                return p
    for pre in ('ir_', 'IR_', 'd_', 'depth_', 'dep_', 'DEP_'):
        for ext in IMG_EXTS:
            p = d / (pre + stem + ext)
            if p.is_file():
                return p
    return None


def encode_depth(d, max_depth=19999.0):
    """uint16 毫米深度 -> (H,W,3)：[线性, 逆深度(近=亮), 有效掩码]。0 视为无效。"""
    d = np.asarray(d, dtype=np.float32)
    valid = d > 0
    if not valid.any():
        return np.zeros(d.shape + (3,), dtype=np.uint8)
    fill = float(np.median(d[valid]))
    df = np.where(valid, d, fill)
    lin = np.clip(df / float(max_depth), 0.0, 1.0)
    lin8 = (lin * 255.0).astype(np.uint8)
    return np.stack([lin8, (255 - lin8).astype(np.uint8),
                     (valid * 255).astype(np.uint8)], axis=-1)


PASTE_CLASSES = tuple(int(x) for x in os.environ.get("V5_PASTE_CLASSES", "7,11").split(","))
# 论文 04 启示：往最难的类贴。默认 ball(7),tricycle(11)；可设 "7,11,10,1" 加 uav,boat


N_CH = 9
MAX_DEPTH_MM = 19999.0


def _thermal_saliency(ir, bgr):
    """(IR - luma(RGB)) / 2 + 128 -> uint8.  128 = neutral, >128 = hot&dark."""
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    diff = ir.astype(np.int16) - gray.astype(np.int16)
    return np.clip(diff // 2 + 128, 0, 255).astype(np.uint8)


def _depth_edge(depth_inv):
    """Sobel magnitude of the inverse-depth map (near = bright), normalised."""
    gx = cv2.Sobel(depth_inv, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(depth_inv, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.magnitude(gx, gy)
    m = float(mag.max())
    if m > 1e-6:
        mag = mag * (255.0 / m)
    return np.clip(mag, 0.0, 255.0).astype(np.uint8)


def encode_depth_v5(depth, max_depth=MAX_DEPTH_MM):
    """encode_depth() with a dtype branch for the 8-bit depth subset.

    1851/2000 depth files are uint16 millimetres and use the v4 encoder
    unchanged.  The other 149 are 8-bit, already-normalised JPEGs; the
    millimetre path maps them to d/19999*255 ~ 0..3, i.e. near-black.  They
    are passed through instead:

        ch4 = d            ch5 = 255 - d          ch6 = (d > 0) * 255

    Assumption: 0 means "no measurement" (same convention as the uint16
    files) and the 8-bit scale increases monotonically with distance.  Invalid
    pixels are filled with the median of the valid ones, exactly like the
    uint16 path, so the network never learns "0 == infinitely far".
    V5_DROP_8BIT_DEPTH=1 switches to the conservative alternative (pretend
    the geometry is missing for those images).
    """
    d = np.asarray(depth)
    if d.dtype != np.uint8:
        return encode_depth(d, max_depth=max_depth)
    if os.environ.get("V5_DROP_8BIT_DEPTH", "0") == "1":
        return np.zeros(d.shape[:2] + (3,), dtype=np.uint8)
    valid = d > 0
    if not valid.any():
        return np.zeros(d.shape[:2] + (3,), dtype=np.uint8)
    filled = np.where(valid, d, int(np.median(d[valid]))).astype(np.uint8)
    return np.stack([filled, (255 - filled).astype(np.uint8),
                     (valid * 255).astype(np.uint8)], axis=-1)


PHOTOMETRIC_ENV = "V5_PHOTOMETRIC"


def _hsv_lut(bgr_u8, gains):
    """Ultralytics-style hue/sat/val LUT jitter on a 3-channel uint8 image."""
    x = np.arange(0, 256, dtype=np.float32)
    lut_hue = ((x + gains[0] * 180.0) % 180.0).astype(np.uint8)
    lut_sat = np.clip(x * (gains[1] + 1.0), 0, 255).astype(np.uint8)
    lut_val = np.clip(x * (gains[2] + 1.0), 0, 255).astype(np.uint8)
    lut_sat[0] = 0
    hue, sat, val = cv2.split(cv2.cvtColor(bgr_u8, cv2.COLOR_BGR2HSV))
    merged = cv2.merge((cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat),
                        cv2.LUT(val, lut_val)))
    return cv2.cvtColor(merged, cv2.COLOR_HSV2BGR)


def photometric_jitter9(img):
    """Photometric jitter for the 9-channel contract (in place, uint8).

    ch0..3 (BGR + IR) are jittered, ch7 is recomputed from the jittered pair,
    ch4/ch5/ch6/ch8 stay bit-identical.  Env knobs:
        V5_PHOTOMETRIC=0   disable entirely
        V5_HSV_H/S/V       hue/sat/val gains (0.015 / 0.5 / 0.3)
        V5_IR_GAIN         IR brightness gain (0.3)
    """
    if os.environ.get(PHOTOMETRIC_ENV, "1") != "1":
        return img
    h = float(os.environ.get("V5_HSV_H", "0.015"))
    s = float(os.environ.get("V5_HSV_S", "0.5"))
    v = float(os.environ.get("V5_HSV_V", "0.3"))
    gain_ir = float(os.environ.get("V5_IR_GAIN", "0.3"))

    rnd = np.array([random.uniform(-1.0, 1.0) for _ in range(3)],
                   dtype=np.float32)
    gains = rnd * np.array([h, s, v], dtype=np.float32)
    if bool(np.any(gains)):
        img[:, :, :3] = _hsv_lut(np.ascontiguousarray(img[:, :, :3]), gains)

    gain = 1.0 + random.uniform(-1.0, 1.0) * gain_ir
    if abs(gain - 1.0) > 1e-6:
        lut = np.clip(np.arange(0, 256, dtype=np.float32) * gain,
                      0, 255).astype(np.uint8)
        img[:, :, 3] = cv2.LUT(np.ascontiguousarray(img[:, :, 3]), lut)

    # 错位抖动增强（自动化学报综述的 RoI-Jitter 思想）：真实多模态数据只是"弱对齐"，
    # 训练时把 ch3..8 随机平移 ±3px（RGB 与标签不动），教融合模块对不齐也能用。
    # 开关：V5_MODAL_SHIFT=概率（建议 0.25）
    if os.environ.get("V5_MODAL_SHIFT", "0") not in ("0", "") \
            and random.random() < float(os.environ["V5_MODAL_SHIFT"]):
        h0, w0 = img.shape[:2]
        dx, dy = random.randint(-3, 3), random.randint(-3, 3)
        if dx or dy:
            M = np.float32([[1, 0, dx], [0, 1, dy]])
            img[:, :, 3:] = cv2.warpAffine(img[:, :, 3:], M, (w0, h0),
                                           borderMode=cv2.BORDER_REPLICATE)

    # 暗场景增强（夜间缺失的补救，论文级 trick）：
    # 以 V5_DARKEN 概率把 RGB 压暗到 [LO,HI] 倍（红外/深度物理上不受光照影响，保持原样），
    # 人为制造"RGB 弱、红外强"的训练对，教辅助支路在低光条件依赖热信息。
    p_dark = float(os.environ.get("V5_DARKEN", "0"))
    if p_dark > 0 and random.random() < p_dark:
        g = random.uniform(float(os.environ.get("V5_DARKEN_LO", "0.35")),
                           float(os.environ.get("V5_DARKEN_HI", "0.75")))
        img[:, :, :3] = np.clip(img[:, :, :3].astype(np.float32) * g, 0, 255).astype(np.uint8)

    # 概率灰度化（等价 ultralytics grayscale 增强；只动 RGB，红外/深度不动，
    # ch7 热显著性基于 luma(RGB)，RGB 变了必须重算）
    if random.random() < float(os.environ.get("V5_GRAYSCALE", "0.15")):
        gray = cv2.cvtColor(np.ascontiguousarray(img[:, :, :3]), cv2.COLOR_BGR2GRAY)
        img[:, :, :3] = cv2.merge([gray, gray, gray])

    img[:, :, 7] = _thermal_saliency(img[:, :, 3],
                                     np.ascontiguousarray(img[:, :, :3]))
    return img


def _enhance_ir(ir):
    """红外输入增强（朋友的建议：官方 IR 数据质量差，先处理再训练）。
    2%~98% 自动对比度拉伸 + CLAHE 自适应均衡 —— 热成像检测的标准预处理。
    开关：V5_IR_ENH=1。load_tri 同时服务训练/验证/提交，三端自动一致。
    """
    if os.environ.get("V5_IR_ENH", "0") != "1":
        return ir
    lo, hi = np.percentile(ir, 2), np.percentile(ir, 98)
    if hi > lo + 1:
        ir = np.clip((ir.astype(np.float32) - lo) * 255.0 / (hi - lo), 0, 255).astype(np.uint8)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(ir)


def _enhance_depth(d):
    """深度输入增强：中值滤波去 JPEG 噪点（8 位图受益最大）。开关：V5_DEPTH_ENH=1。"""
    if os.environ.get("V5_DEPTH_ENH", "0") != "1":
        return d
    if d.dtype == np.uint8:
        return cv2.medianBlur(d, 3)
    return cv2.medianBlur(d, 3)   # uint16 同样支持


def _val_depth_quirk(img):
    """复刻老 base.py 验证期的行为：dp<10 的像素三通道全部设为 104（uint8 回绕值）。

    你的 y26x_o365 在验证/提交时深度通道经历过这个处理，为了让它在我们
    管线下的行为与拿 48 分时完全一致，验证和推理路径都要复刻。
    """
    dlin = img[:, :, 4]
    m = dlin < 10
    if m.any():
        img[:, :, 4][m] = 104
        img[:, :, 5][m] = 104
        img[:, :, 6][m] = 104
    return img


def load_tri(rgb_path, ir_path, depth_path, max_depth=MAX_DEPTH_MM, val_quirk=False):
    """Read the three modalities into one HxWx9 uint8 array (first 3 = BGR)."""
    rgb = cv2.imread(str(rgb_path))
    if rgb is None:
        raise FileNotFoundError(rgb_path)
    h0, w0 = rgb.shape[:2]

    ir = cv2.imread(str(ir_path), cv2.IMREAD_GRAYSCALE)
    if ir is None:
        ir = np.zeros((h0, w0), dtype=np.uint8)
    else:
        ir = cv2.resize(ir, (w0, h0), interpolation=cv2.INTER_LINEAR)

    d = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
    if d is None:
        d = np.zeros((h0, w0), dtype=np.uint16)
    if d.ndim == 3:
        d = d[:, :, 0]
    d = cv2.resize(d, (w0, h0), interpolation=cv2.INTER_NEAREST)
    ir = _enhance_ir(ir)
    d = _enhance_depth(d)
    d3 = encode_depth_v5(d, max_depth=max_depth)       # (H,W,3) lin/inv/valid
    # YT-SWIN 论文启发（LSD 频带思想）：红外的价值在低频热斑。
    # V5_IR_LOW=1 时用"高斯低通红外"替换贡献最弱的 ch5（深度逆通道，消融依赖度~0.0005）
    if os.environ.get("V5_IR_LOW", "0") == "1":
        d3[:, :, 1] = cv2.GaussianBlur(ir, (9, 9), 2)

    sal = _thermal_saliency(ir, rgb)
    edge = _depth_edge(d3[:, :, 1])

    img = np.concatenate([rgb, ir[..., None], d3,
                          sal[..., None], edge[..., None]], axis=-1)
    img = img.astype(np.uint8)
    if val_quirk:
        img = _val_depth_quirk(img)
    return img, (h0, w0)


class TriModalDatasetV4(YOLODataset):
    """9-channel RGB+TIR+Depth dataset (same class name as v4 for drop-in use).

    Set the four class attributes (TRAIN_IR / VAL_IR / TRAIN_DEPTH / VAL_DEPTH)
    to the ir/ and depth/ folders before training.
    """

    TRAIN_IR = None
    VAL_IR = None
    TRAIN_DEPTH = None
    VAL_DEPTH = None
    MODAL_DROP_ENV = "V5_MODAL_DROP"
    DROP_IR_ENV = "V5_DROP_IR"
    DROP_DEPTH_ENV = "V5_DROP_DEPTH"

    def __init__(self, *args, max_depth=MAX_DEPTH_MM, **kwargs):
        super().__init__(*args, **kwargs)
        self.max_depth = max_depth
        def _as_dirs(x):   # 支持目录列表（val 里混有来自 train 目录的图）
            return [Path(d) for d in x] if isinstance(x, (list, tuple)) else Path(x)
        if self.augment:
            self.ir_dir = _as_dirs(self.TRAIN_IR)
            self.depth_dir = _as_dirs(self.TRAIN_DEPTH)
        else:
            self.ir_dir = _as_dirs(self.VAL_IR)
            self.depth_dir = _as_dirs(self.VAL_DEPTH)
        valid, self.ir_paths, self.depth_paths = [], {}, {}
        for idx, im_path in enumerate(self.im_files):
            stem = Path(im_path).stem
            ir_path = _find_modal_any(self.ir_dir, stem)
            depth_path = _find_modal_any(self.depth_dir, stem)
            if ir_path is not None and depth_path is not None:
                valid.append(idx)
                self.ir_paths[stem] = ir_path
                self.depth_paths[stem] = depth_path
        self.im_files = [self.im_files[i] for i in valid]
        self.labels = [self.labels[i] for i in valid]
        self.buffer = list(range(len(self.im_files)))
        print("[TriModalDatasetV5] valid samples: %d (%dch)" % (len(self.im_files), N_CH))

    def load_image(self, i, rect_mode=True):
        im_path = self.im_files[i]
        stem = Path(im_path).stem
        img, (h0, w0) = load_tri(im_path, self.ir_paths[stem],
                                 self.depth_paths[stem], self.max_depth,
                                 val_quirk=not self.augment)

        if self.augment:
            photometric_jitter9(img)
            _md = float(os.environ.get(self.MODAL_DROP_ENV, "0.25"))
            if _md > 0.0:
                if random.random() < _md:
                    img[:, :, 3] = 0
                    img[:, :, 7] = 128
                if random.random() < _md:
                    img[:, :, 4:7] = 0
                    img[:, :, 8] = 0
        if os.environ.get(self.DROP_IR_ENV, "0") == "1":
            img[:, :, 3] = 0
            img[:, :, 7] = 128
        if os.environ.get(self.DROP_DEPTH_ENV, "0") == "1":
            img[:, :, 4:7] = 0
            img[:, :, 8] = 0

        try:
            imsz = int(self.imgsz)
        except Exception:
            imsz = 0
        if imsz > 0:
            if rect_mode:
                if imsz != max(h0, w0):
                    r = imsz / max(h0, w0)
                    w, h = (min(math.ceil(w0 * r), imsz), min(math.ceil(h0 * r), imsz))
                    img = cv2.resize(img, (w, h), interpolation=cv2.INTER_LINEAR)
            elif not (h0 == w0 == imsz):
                img = cv2.resize(img, (imsz, imsz), interpolation=cv2.INTER_LINEAR)
        return img, (h0, w0), img.shape[:2]


class RarePasteDatasetV4(TriModalDatasetV4):
    """Tri-modal copy-paste of rare classes (patch copied in all 9 channels)."""

    PASTE_CLASSES = PASTE_CLASSES
    PASTE_PROB = 0.5

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.rare_pool = []
        for idx, lb in enumerate(self.labels):
            cls = lb.get("cls")
            if cls is None or len(cls) == 0:
                continue
            if set(int(c) for c in cls) & set(self.PASTE_CLASSES):
                self.rare_pool.append(idx)
        print("[RarePasteDatasetV5] rare-paste pool: %d images" % len(self.rare_pool))

    def get_image_and_label(self, index):
        label = deepcopy(self.labels[index])
        label.pop("shape", None)
        label["img"], label["ori_shape"], label["resized_shape"] = self.load_image(index)
        label["ratio_pad"] = (
            label["resized_shape"][0] / label["ori_shape"][0],
            label["resized_shape"][1] / label["ori_shape"][1],
        )
        if self.rect:
            label["rect_shape"] = self.batch_shapes[self.batch[index]]
        if self.augment and self.rare_pool and random.random() < self.PASTE_PROB:
            self._paste_rare(label)
        return self.update_labels_info(label)

    def _paste_rare(self, label):
        img = label["img"]
        h, w = img.shape[:2]
        bboxes = list(np.asarray(label["bboxes"], dtype=np.float32))
        cls = [float(x) for x in np.asarray(label["cls"], dtype=np.float32).reshape(-1)]

        for _ in range(random.randint(1, 2)):
            src_idx = random.choice(self.rare_pool)
            src = self.labels[src_idx]
            src_cls = np.asarray(src["cls"])
            src_box = np.asarray(src["bboxes"])
            pos = [i for i, c in enumerate(src_cls) if int(c) in self.PASTE_CLASSES]
            if not pos:
                continue
            pick = random.choice(pos)
            simg, (sh, sw), _ = self.load_image(src_idx)
            cx, cy, bw, bh = [float(v) for v in src_box[pick]]
            x1 = int(round((cx - bw / 2) * sw)); y1 = int(round((cy - bh / 2) * sh))
            x2 = int(round((cx + bw / 2) * sw)); y2 = int(round((cy + bh / 2) * sh))
            if x2 <= x1 or y2 <= y1:
                continue
            patch = simg[max(y1, 0):y2, max(x1, 0):x2].copy()
            if patch.size == 0:
                continue
            scale = random.uniform(0.6, 1.5)
            pw = max(8, int(round(patch.shape[1] * scale)))
            ph = max(8, int(round(patch.shape[0] * scale)))
            pw = min(pw, w); ph = min(ph, h)
            if pw < 8 or ph < 8:
                continue
            patch = cv2.resize(patch, (pw, ph), interpolation=cv2.INTER_LINEAR)
            px = random.randint(0, max(0, w - pw))
            py = random.randint(0, max(0, h - ph))
            img[py:py + ph, px:px + pw] = patch
            bboxes.append((float((px + pw / 2) / w), float((py + ph / 2) / h),
                           float(pw / w), float(ph / h)))
            cls.append(int(src_cls[pick]))

        if bboxes:
            label["bboxes"] = np.asarray(bboxes, dtype=np.float32)
            label["cls"] = np.asarray(cls, dtype=np.float32).reshape(-1, 1)
