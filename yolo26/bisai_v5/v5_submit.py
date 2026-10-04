# AUTO-GENERATED from v4_submit.py by bisai_v5/_make_v5_scripts.py -- edit the v4
# source and re-run the generator (or edit here and re-run)

# -*- coding: utf-8 -*-


import argparse
import os
import shutil
import tempfile
import zipfile
from pathlib import Path

import numpy as np
import torch
import cv2

BASE_DIR = Path(__file__).resolve().parent
import sys
sys.path.insert(0, str(BASE_DIR.parent))

from ultralytics import YOLO                                # noqa: E402
from v5_model import N_CH                                   # noqa: E402
from v5_dataset import load_tri, _find_modal                # noqa: E402
from v5_model import apply_fusion_v5 as apply_fusion_v4      # noqa: E402


# 同 v4_submit：与评估/调优路径的 max_det=300 对齐，避免密集图静默丢框。
MAX_DET = 300

CONF_TH = [0.35, 0.15, 0.35, 0.30, 0.30, 0.30,
           0.35, 0.10, 0.35, 0.20, 0.20, 0.10]     # per-class conf (12)
NMS_IOU = [0.5,  0.5,  0.5,  0.5,  0.5,  0.5,
           0.5,  0.55, 0.5,  0.5,  0.55, 0.55]    # per-class NMS iou


def build_map(d):
    if not d.is_dir():
        return {}
    return {p.stem: str(p) for p in d.iterdir()
            if p.suffix.lower() in (".png", ".jpg", ".jpeg")}


def make_tiles(img, grid=2, overlap=0.25):
    """把 HWC 图切成 grid×grid 个重叠小块，返回 [(x0, y0, tile_img), ...]。"""
    h, w = img.shape[:2]
    th, tw = max(1, h // grid), max(1, w // grid)
    sy, sx = max(1, int(th * (1 - overlap))), max(1, int(tw * (1 - overlap)))
    ys, xs = [0], [0]
    while ys[-1] + th < h:
        ys.append(min(ys[-1] + sy, h - th))
    while xs[-1] + tw < w:
        xs.append(min(xs[-1] + sx, w - tw))
    return [(x, y, img[y:y + th, x:x + tw].copy()) for y in ys for x in xs]


def dark_enhance_rgb(img):
    """仅暗图（均值<70）对 RGB 三通道做 CLAHE 提亮，红外/深度不动。"""
    if img[:, :, :3].mean() >= 70:
        return img
    out = img.copy()
    for c in range(3):
        out[:, :, c] = cv2.createCLAHE(2.0, (8, 8)).apply(out[:, :, c])
    return out


def letterbox_multi(im, new=728, pad=(114, 0, 0)):
    """Letterbox a HxWxC image with per-channel pad values -> (img, scale, (dw,dh))."""
    if isinstance(new, int):
        new = (new, new)

    s = im.shape[:2]
    scale = min(new[0] / s[0], new[1] / s[1])
    nu = (int(round(s[1] * scale)), int(round(s[0] * scale)))
    dw, dh = (new[1] - nu[0]) / 2, (new[0] - nu[1]) / 2

    if nu != s[::-1]:
        im = cv2.resize(im, nu, interpolation=cv2.INTER_LINEAR)

    t, b = int(np.floor(dh)), int(np.ceil(dh))
    l, r = int(np.floor(dw)), int(np.ceil(dw))
    c = im.shape[2]
    pads = []
    for ch in range(c):
        v = pad[ch] if ch < len(pad) else 0
        pads.append(np.pad(im[..., ch], ((t, b), (l, r)), mode="constant", constant_values=v))

    return np.stack(pads, axis=-1), scale, (dw, dh)


def nms_merge(dets, iou_th=0.5, iou_table=None):
    if len(dets) == 0:
        return np.empty((0, 6))

    if iou_table is not None:
        n_cls = len(iou_table)
        dets = dets[(dets[:, 5] >= 0) & (dets[:, 5] < n_cls)]
        if len(dets) == 0:
            return np.empty((0, 6))

    out = []
    for c in np.unique(dets[:, 5].astype(int)):
        th = iou_th if iou_table is None else iou_table[int(c)]
        m = dets[dets[:, 5].astype(int) == c]
        m = m[np.argsort(-m[:, 4])]
        keep = np.ones(len(m), bool)

        for i in range(len(m)):
            if not keep[i]:
                continue
            for j in range(i + 1, len(m)):
                if not keep[j]:
                    continue
                x1 = max(m[i, 0], m[j, 0]); y1 = max(m[i, 1], m[j, 1])
                x2 = min(m[i, 2], m[j, 2]); y2 = min(m[i, 3], m[j, 3])
                if x2 > x1 and y2 > y1:
                    inter = (x2 - x1) * (y2 - y1)
                    u = ((m[i, 2] - m[i, 0]) * (m[i, 3] - m[i, 1]) +
                         (m[j, 2] - m[j, 0]) * (m[j, 3] - m[j, 1]) - inter)
                    if inter / u > th:
                        keep[j] = False

        out.append(m[keep])

    return np.vstack(out) if out else np.empty((0, 6))


def _iou(a, b):
    x1 = max(a[0], b[0]); y1 = max(a[1], b[1])
    x2 = min(a[2], b[2]); y2 = min(a[3], b[3])
    if x2 <= x1 or y2 <= y1:
        return 0.0
    inter = (x2 - x1) * (y2 - y1)
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / max(u, 1e-9)


def wbf_merge(dets, iou_th=0.5, iou_table=None):
    """Weighted box fusion: confidence-weighted box average, max confidence."""
    if len(dets) == 0:
        return np.empty((0, 6))

    if iou_table is not None:
        n_cls = len(iou_table)
        dets = dets[(dets[:, 5] >= 0) & (dets[:, 5] < n_cls)]
        if len(dets) == 0:
            return np.empty((0, 6))

    out = []
    for c in np.unique(dets[:, 5].astype(int)):
        th = iou_th if iou_table is None else iou_table[int(c)]
        m = dets[dets[:, 5].astype(int) == c]
        m = m[np.argsort(-m[:, 4])]
        clusters = []

        for box in m:
            best, best_iou = None, -1.0
            for cl in clusters:
                v = _iou(box, cl["box"])
                if v >= th and v > best_iou:
                    best, best_iou = cl, v

            if best is not None:
                wsum = best["w"] + float(box[4])
                best["box"] = (best["box"] * best["w"] + box[:4] * float(box[4])) / wsum
                best["w"] = wsum
                best["conf"] = max(best["conf"], float(box[4]))
            else:
                clusters.append({"box": box[:4].astype(np.float64).copy(),
                                 "w": float(box[4]), "conf": float(box[4])})

        for cl in clusters:
            out.append([cl["box"][0], cl["box"][1], cl["box"][2], cl["box"][3],
                        cl["conf"], float(c)])

    res = np.asarray(out, dtype=np.float64)
    return res[np.argsort(-res[:, 4])]


def write_dets(st, dets, h0, w0, conf_th):
    if len(dets) == 0:
        st.touch()
        return

    keep = [d for d in dets
            if 0 <= int(d[5]) < len(conf_th) and d[4] >= conf_th[int(d[5])]]

    if not keep:
        st.touch()
        return

    keep = np.array(keep)
    keep = keep[np.argsort(-keep[:, 4])][:MAX_DET]

    cx = np.clip((keep[:, 0] + keep[:, 2]) / 2 / w0, 0, 1)
    cy = np.clip((keep[:, 1] + keep[:, 3]) / 2 / h0, 0, 1)
    wn = np.clip((keep[:, 2] - keep[:, 0]) / w0, 0, 1)
    hn = np.clip((keep[:, 3] - keep[:, 1]) / h0, 0, 1)

    with open(st, "w", newline="") as f:
        for i in range(len(keep)):
            f.write(f"{int(keep[i, 5])} {cx[i]:.6f} {cy[i]:.6f} {wn[i]:.6f} {hn[i]:.6f} {keep[i, 4]:.6f}\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model_pt", nargs="?", default=None)
    ap.add_argument("--test-dir", default=os.environ.get("TEST_DIR", ""))
    ap.add_argument("--out", default=os.environ.get("V4_OUT", "/root/autodl-tmp/submit_v4"))
    ap.add_argument("--merge", choices=["nms", "wbf"], default="nms")
    ap.add_argument("--scales", default=os.environ.get("V4_SCALES", "728"),
                    help="inference size, single scale only (default: 728)")
    ap.add_argument("--flip", action="store_true", default=False,
                    help="TTA flip is disabled by default for conservative submission")
    ap.add_argument("--tile", type=int, default=int(os.environ.get("V5_TILE", "0")),
                    help="切片推理（SAHI 思想）：切成 N×N 重叠小块分别检测后与整图结果合并，"
                         "小目标相对变大。0=关闭，推荐 2")
    ap.add_argument("--tile-overlap", type=float, default=0.25)
    ap.add_argument("--rgb-dark-enhance", action="store_true",
                    default=os.environ.get("V5_RGB_DARK", "0") == "1",
                    help="测试时只对暗图像的 RGB 做 CLAHE 提亮（夜间测试场景专用）")
    ap.add_argument("--legacy", action="store_true",
                    help="老9通道模型直接提交：挂融合模块+warmup=0（行为严格等于原模型）")
    ap.add_argument("--conf", type=float, default=0.001)
    ap.add_argument("--conf-file", default=os.environ.get("V4_CONF_FILE", ""),
                    help="json produced by v4_tune.py (conf_th / nms_iou)")
    args = ap.parse_args()

    import v5_channels
    v5_channels.install(verbose=False)  # 稳健版 9 通道适配

    if args.conf_file and Path(args.conf_file).exists():
        global CONF_TH, NMS_IOU
        import json as _json
        with open(args.conf_file, encoding="utf-8") as _f:
            _j = _json.load(_f)
        if isinstance(_j, dict) and _j.get("conf_th"):
            CONF_TH = [float(x) for x in _j["conf_th"]]
            print("[conf-file] conf_th loaded from", args.conf_file)
        if isinstance(_j, dict) and _j.get("nms_iou"):
            NMS_IOU = [float(x) for x in _j["nms_iou"]]
            print("[conf-file] nms_iou loaded from", args.conf_file)

    out_dir = Path(args.out)
    zip_path = out_dir.with_suffix(".zip")

    # locate test data
    test_base = None
    if args.test_dir and Path(args.test_dir).is_dir():
        test_base = Path(args.test_dir)
    else:
        for c in ["/root/autodl-tmp/test", "/root/autodl-tmp/text2/test",
                  "/root/autodl-tmp/text2", "test"]:
            if Path(c).is_dir():
                test_base = Path(c)
                break

    if test_base is None or not (test_base / "images").is_dir():
        print("[error] test data not found (--test-dir)")
        return

    rgb_dir, ir_dir, dep_dir = (test_base / "images"), (test_base / "ir"), (test_base / "depth")


    scales = [int(x) for x in str(args.scales).split(',') if x.strip()]

    # model
    mp = Path(args.model_pt) if args.model_pt else None
    if mp is None or not mp.exists():
        cands = sorted(Path("runs/train").glob("*/weights/best.pt"))
        cands += sorted(Path("runs/detect/runs/train").glob("*/weights/best.pt"))
        mp = cands[-1] if cands else None

    if mp is None:
        print("[error] no model checkpoint found")
        return

    print("[model]", mp)

    model = YOLO(str(mp))
    is_fused = hasattr(model.model, "aux_encoder") or hasattr(model.model, "inject")
    if args.legacy and not is_fused:
        from v5_model import set_inject_warmup
        apply_fusion_v4(model.model)
        set_inject_warmup(model.model, 0.0)
        is_fused = True
        print("[legacy] 已挂融合模块+warmup=0，行为等价于原9通道模型")
    if is_fused:
        apply_fusion_v4(model.model)
        from v5_model import set_inject_warmup
        set_inject_warmup(model.model, 1.0)   # 推理时注入全开

    print(f"[inference] scales={scales} (single scale 728), flip={args.flip}, merge={args.merge}")

    ir_map, de_map = build_map(ir_dir), build_map(dep_dir)
    rgb_files = sorted([p for p in rgb_dir.iterdir() if p.suffix.lower() in (".png", ".jpg", ".jpeg")])
    print(f"[test images] {len(rgb_files)}")

    if out_dir.exists():
        shutil.rmtree(str(out_dir))
    out_dir.mkdir(parents=True)

    tmp = Path(tempfile.mkdtemp(prefix="v5_submit_"))
    miss = 0

    for idx, rgb_p in enumerate(rgb_files):
        stem = rgb_p.stem
        st = out_dir / f"{stem}.txt"

        if is_fused and (stem not in ir_map or stem not in de_map):
            miss += 1
            st.touch()
            continue

        if is_fused:
            img, (h0, w0) = load_tri(rgb_p, ir_map[stem], de_map[stem], val_quirk=True)
            img[:, :, :3] = img[:, :, 2::-1]        # BGR -> RGB for the RGB part
            if args.rgb_dark_enhance:
                img = dark_enhance_rgb(img)

            all_dets = []
            # 整图 + 切片并行（SAHI 已知风险：切片会伤大目标，整图结果兜底），NMS 合并
            regions = [(0, 0, img)] + (make_tiles(img, args.tile, args.tile_overlap) if args.tile else [])
            for (gx, gy, reg) in regions:
                for s in scales:
                    lb, scale, (dw, dh) = letterbox_multi(reg, new=s, pad=(114, 0, 0))
                    t = torch.from_numpy(lb).permute(2, 0, 1).contiguous().float() / 255.0
                    t = t.unsqueeze(0).to("cuda:0" if torch.cuda.is_available() else "cpu")

                    # 切片模式下 gx/gy 是小块在原图的偏移，预测框映射回去再合并
                    for tt in (t, t.flip(-1)) if args.flip else (t,):
                        r = model(tt, verbose=False, max_det=MAX_DET, conf=args.conf)
                        if r[0].boxes is None or len(r[0].boxes) == 0:
                            continue

                        xy = r[0].boxes.xyxy.cpu().numpy().copy()
                        conf = r[0].boxes.conf.cpu().numpy()[:, None]
                        cls = r[0].boxes.cls.cpu().numpy()[:, None]

                        if tt is not t:
                            W = t.shape[3]
                            xy[:, [0, 2]] = W - xy[:, [2, 0]]

                        xy[:, [0, 2]] = (xy[:, [0, 2]] - dw) / scale + gx
                        xy[:, [1, 3]] = (xy[:, [1, 3]] - dh) / scale + gy
                        all_dets.append(np.hstack([xy, conf, cls]))
        else:
            rgb_bgr = cv2.imread(str(rgb_p))
            h0, w0 = rgb_bgr.shape[:2]
            img_rgb = rgb_bgr[:, :, ::-1]

            all_dets = []
            for s in scales:
                lb, scale, (dw, dh) = letterbox_multi(img_rgb, new=s, pad=(114,))
                t = torch.from_numpy(lb).permute(2, 0, 1).contiguous().float() / 255.0
                t = t.unsqueeze(0).to("cuda:0" if torch.cuda.is_available() else "cpu")

                # 取消 TTA 多尺度；即使开启 flip，也只是单尺度下的可选翻转
                for tt in (t, t.flip(-1)) if args.flip else (t,):
                    r = model(tt, verbose=False, max_det=MAX_DET, conf=args.conf)
                    if r[0].boxes is None or len(r[0].boxes) == 0:
                        continue

                    xy = r[0].boxes.xyxy.cpu().numpy().copy()
                    conf = r[0].boxes.conf.cpu().numpy()[:, None]
                    cls = r[0].boxes.cls.cpu().numpy()[:, None]

                    if tt is not t:
                        W = t.shape[3]
                        xy[:, [0, 2]] = W - xy[:, [2, 0]]

                    xy[:, [0, 2]] = (xy[:, [0, 2]] - dw) / scale + gx
                    xy[:, [1, 3]] = (xy[:, [1, 3]] - dh) / scale + gy
                    all_dets.append(np.hstack([xy, conf, cls]))

        if all_dets:
            stack = np.vstack(all_dets)
            dets = (wbf_merge(stack, 0.5, NMS_IOU) if args.merge == "wbf"
                    else nms_merge(stack, 0.5, NMS_IOU))
        else:
            dets = np.empty((0, 6))

        write_dets(st, dets, h0, w0, CONF_TH)

        if (idx + 1) % 50 == 0:
            print(f"  [{idx + 1}/{len(rgb_files)}]")

    shutil.rmtree(str(tmp), ignore_errors=True)
    print(f"[done] {len(rgb_files)} files, {miss} missing ir/depth")

    with zipfile.ZipFile(str(zip_path), "w", zipfile.ZIP_DEFLATED) as zf:
        for t in sorted(out_dir.glob("*.txt")):
            zf.write(t, arcname=t.name)

    print(f"[zip] {zip_path} ({zip_path.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
