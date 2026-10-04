# -*- coding: utf-8 -*-
"""v5_tune -- 逐类置信度标定。

在 val 集上扫每个类别的 conf 阈值，最大化 F1（兼顾 precision/recall），
输出 JSON 供 v5_submit.py --conf-file 直接使用。

注意：你的 val 泄露进了训练，绝对数值偏高，但"哪类该压 conf、哪类该抬"
的相对结论仍然有效（两个候选模型共用同一份 val 对比时尤其可靠）。

用法：
  python bisai_v5/v5_tune.py \
      --model runs/detect/runs/train/v5_fusion_b/weights/best.pt \
      --data /root/autodl-tmp/data_full --imgsz 1024 \
      --out /root/autodl-tmp/tuned_conf.json
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

NAMES = ["person", "boat", "animal", "seat", "sign", "bicycle",
         "car", "ball", "light", "garbage_can", "uav", "tricycle"]
NC = 12


def xywhn_to_xyxy(b, w, h):
    cx, cy, bw, bh = b
    return [max(0.0, (cx - bw / 2) * w), max(0.0, (cy - bh / 2) * h),
            min(w, (cx + bw / 2) * w), min(h, (cy + bh / 2) * h)]


def iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    if x2 <= x1 or y2 <= y1:
        return 0.0
    inter = (x2 - x1) * (y2 - y1)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / max(ua, 1e-9)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", default="/root/autodl-tmp/data_full")
    ap.add_argument("--imgsz", type=int, default=1024)
    ap.add_argument("--conf", type=float, default=0.001)
    ap.add_argument("--flip", action="store_true", help="TTA 翻转合并")
    ap.add_argument("--out", default="/root/autodl-tmp/tuned_conf.json")
    args = ap.parse_args()

    import v5_channels
    v5_channels.install()

    import cv2
    from ultralytics import YOLO
    from v5_model import apply_fusion_v5, set_inject_warmup
    from v5_dataset import load_tri, _find_modal
    from v5_submit import letterbox_multi   # 复用同一套 letterbox，保证口径一致

    data = Path(args.data)
    model = YOLO(args.model)
    if hasattr(model.model, "inject") or hasattr(model.model, "aux_encoder"):
        apply_fusion_v5(model.model)
        set_inject_warmup(model.model, 1.0)
    model.model.cuda().eval()

    # 收集 val 预测与标签
    dets = [[] for _ in range(NC)]        # per class: list of (conf, img_idx, matched_flag)
    gts = [[] for _ in range(NC)]         # per class: list of (img_idx, box)
    val_list = (data / "splits" / "val.txt").read_text().splitlines()
    val_list = [l for l in val_list if l.strip()]

    for idx, line in enumerate(val_list):
        rgb_p = Path(line.strip())
        stem = rgb_p.stem
        ir_dirs = [d for d in (data / "ir_val", data / "ir") if d.is_dir()]
        dp_dirs = [d for d in (data / "depth_val", data / "depth") if d.is_dir()]
        ir_p = None
        for d in ir_dirs:
            ir_p = _find_modal(d, stem)
            if ir_p: break
        dp_p = None
        for d in dp_dirs:
            dp_p = _find_modal(d, stem)
            if dp_p: break
        img, (h0, w0) = load_tri(rgb_p, ir_p, dp_p)
        lb, scale, (dw, dh) = letterbox_multi(img, new=args.imgsz, pad=(114, 0, 0))
        t = torch.from_numpy(lb).permute(2, 0, 1)[None].float().cuda() / 255.0

        boxes_all = []
        with torch.no_grad():
            for tt in (t, t.flip(-1)) if args.flip else (t,):
                r = model(tt, verbose=False, max_det=300, conf=args.conf)[0]
                if r.boxes is None or len(r.boxes) == 0:
                    continue
                xy = r.boxes.xyxy.cpu().numpy().copy()
                cf = r.boxes.conf.cpu().numpy()
                cl = r.boxes.cls.cpu().numpy().astype(int)
                if tt is not t:
                    W = t.shape[3]
                    xy[:, [0, 2]] = W - xy[:, [2, 0]]
                xy[:, [0, 2]] = (xy[:, [0, 2]] - dw) / scale
                xy[:, [1, 3]] = (xy[:, [1, 3]] - dh) / scale
                boxes_all.append(np.hstack([xy, cf[:, None], cl[:, None]]))
        if boxes_all:
            allb = np.vstack(boxes_all)
            for c in range(NC):
                m = allb[allb[:, 5] == c]
                for x1, y1, x2, y2, cf, _ in m:
                    dets[c].append([float(cf), idx, [x1, y1, x2, y2], False])

        # GT（labels 目录与 images 同级替换）
        lab = Path(str(rgb_p).replace("/images/", "/labels/")).with_suffix(".txt")
        if lab.is_file():
            for ln in lab.read_text().splitlines():
                p = ln.split()
                if len(p) >= 5:
                    c = int(p[0])
                    if 0 <= c < NC:
                        gts[c].append([idx, xywhn_to_xyxy([float(v) for v in p[1:5]], w0, h0)])

    # 逐类扫描 conf
    conf_th, nms_iou = [], []
    grid = np.linspace(0.02, 0.90, 89)
    print("%-12s %6s %6s %6s %8s" % ("class", "F1@", "P", "R", "conf"))
    for c in range(NC):
        ds = sorted(dets[c], key=lambda d: -d[0])
        gt = gts[c]
        best = (0.0, 0.25)
        for t in grid:
            used = set()
            tp = 0
            for conf, iidx, box, _ in ds:
                if conf < t:
                    break
                for j, (gidx, gbox) in enumerate(gt):
                    if gidx == iidx and j not in used and iou(box, gbox) >= 0.5:
                        used.add(j); tp += 1
                        break
            fp = sum(1 for d in ds if d[0] >= t) - tp
            fn = len(gt) - tp
            p = tp / max(tp + fp, 1)
            r = tp / max(tp + fn, 1)
            f1 = 2 * p * r / max(p + r, 1e-9)
            if f1 > best[0]:
                best = (f1, t, p, r)
        f1, t, p, r = best
        conf_th.append(round(float(t), 3))
        nms_iou.append(0.5)
        print("%-12s %6.3f %6.3f %6.3f %8.3f" % (NAMES[c], f1, p, r, t))

    out = {"conf_th": conf_th, "nms_iou": nms_iou}
    Path(args.out).write_text(json.dumps(out, indent=1))
    print("\n已写入 %s" % args.out)
    print("提交时加参数：--conf-file %s" % args.out)


if __name__ == "__main__":
    main()
