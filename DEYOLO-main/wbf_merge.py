"""
WBF 模型集成 — 合并两个模型的预测 (exp2 DEYOLO + CityViMD)

输入: 两个预测目录, 每目录每张图一个 txt (格式: class cx cy w h conf, 归一化)
输出: 融合后的提交 txt

用法:
  python wbf_merge.py --dirs predictions_exp2 predictions_cityvimd --out predictions_wbf \
      --weights 1.2 0.8
"""
import argparse
import os
from pathlib import Path

import numpy as np
from ensemble_boxes import weighted_boxes_fusion

NUM_CLASSES = 12


def load_preds(pred_dir):
    """读取一个预测目录 → {stem: (boxes[N,4] xyxy归一化, scores[N], labels[N])}"""
    preds = {}
    for f in Path(pred_dir).glob("*.txt"):
        boxes, scores, labels = [], [], []
        for line in f.read_text().strip().splitlines():
            if not line:
                continue
            cls, cx, cy, w, h, conf = line.split()
            cls, cx, cy, w, h, conf = int(float(cls)), float(cx), float(cy), \
                float(w), float(h), float(conf)
            x1, y1 = cx - w / 2, cy - h / 2
            x2, y2 = cx + w / 2, cy + h / 2
            boxes.append([x1, y1, x2, y2])
            scores.append(conf)
            labels.append(cls)
        preds[f.stem] = (
            np.array(boxes, dtype=np.float32) if boxes else np.zeros((0, 4)),
            np.array(scores, dtype=np.float32) if scores else np.zeros(0),
            np.array(labels, dtype=np.int64) if labels else np.zeros(0, dtype=np.int64),
        )
    return preds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dirs", nargs="+", required=True, help="预测目录列表")
    ap.add_argument("--out", required=True, help="输出目录")
    ap.add_argument("--weights", nargs="+", type=float, default=None,
                    help="各模型 WBF 权重 (默认全1)")
    ap.add_argument("--iou", type=float, default=0.55)
    ap.add_argument("--skip", type=float, default=0.001)
    ap.add_argument("--max-det", type=int, default=100)
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_preds = [load_preds(d) for d in args.dirs]
    weights = args.weights or [1.0] * len(all_preds)
    print(f"模型权重: {weights}")

    # 以第一个目录的文件集合为准 (测试集 1000 张)
    stems = sorted(all_preds[0].keys())
    print(f"样本数: {len(stems)}")

    fused_count = 0
    for stem in stems:
        boxes_list, scores_list, labels_list = [], [], []
        for preds in all_preds:
            b, s, l = preds.get(stem, (np.zeros((0, 4)), np.zeros(0), np.zeros(0)))
            boxes_list.append(b)
            scores_list.append(s)
            labels_list.append(l)

        if sum(len(b) for b in boxes_list) == 0:
            (out_dir / f"{stem}.txt").write_text("")
            continue

        boxes, scores, labels = weighted_boxes_fusion(
            boxes_list, scores_list, labels_list,
            weights=weights, iou_thr=args.iou, skip_box_thr=args.skip)

        # 按分数排序截断
        order = np.argsort(scores)[::-1][:args.max_det]
        boxes, scores, labels = boxes[order], scores[order], labels[order]
        fused_count += len(boxes)

        lines = []
        for box, score, label in zip(boxes, scores, labels):
            x1, y1, x2, y2 = box
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            w, h = x2 - x1, y2 - y1
            lines.append(f"{int(label)} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f} {score:.6f}")
        (out_dir / f"{stem}.txt").write_text("\n".join(lines) + ("\n" if lines else ""))

    print(f"完成! 融合后总框数: {fused_count}, 输出: {out_dir}")


if __name__ == "__main__":
    main()
