"""
prediction_tta.py —— 多尺度 TTA + WBF，冲分专用（最终版）
"""
import os
import cv2
import numpy as np
import torch
from pathlib import Path
from ultralytics import YOLO
from ultralytics.yolo.utils.ops import non_max_suppression
from ensemble_boxes import weighted_boxes_fusion

# ==================== 配置 ====================
weights_path = "my_project/exp_v8x_1280_final/weights/best.pt"
test_dir = Path("test_data")
out_dir = Path("predictions_tta")
imgsz = 1280

# TTA 尺度：1280（训练尺度）+ 1536（放大，小目标更清晰）
# 如果显存/时间紧，只保留 [1280]，但会掉 1~2 分
tta_scales = [1280, 1536]

conf_thres = 0.001      # TTA 时放低，WBF 会过滤噪声
iou_thres = 0.45        # NMS 用，WBF 前预过滤
max_det = 100           # 比赛规则：最多 100 框
# =============================================

CLASS_NAMES = ['person','boat','animal','seat','sign','bicycle',
               'car','ball','light','garbage_can','uav','tricycle']


def letterbox(im, new_shape=1280, color=114):
    shape = im.shape[:2]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
    dw /= 2; dh /= 2
    if shape[::-1] != new_unpad:
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = np.pad(im, ((top, bottom), (left, right), (0, 0)),
                mode='constant', constant_values=color)
    return im, r, (dw, dh)


def preprocess_multiscale(vis_path, ir_path, dep_path, scales):
    """生成多尺度 + 翻转的输入"""
    img_vis = cv2.imread(str(vis_path))
    img_ir = cv2.imread(str(ir_path))
    img_dep = cv2.imread(str(dep_path), cv2.IMREAD_UNCHANGED)
    if img_vis is None:
        raise FileNotFoundError(f"无法读取: {vis_path}")

    h, w = img_vis.shape[:2]
    if img_ir is None:
        img_ir = np.zeros((h, w, 3), dtype=np.uint8)
    if img_dep is None:
        img_dep = np.zeros((h, w), dtype=np.uint16)

    # 深度预处理（必须和训练时完全一致！）
    img_dep = np.clip(img_dep, 0, 20000).astype(np.uint16)
    img_dep = (img_dep / 20000 * 255).astype(np.uint8)
    if len(img_dep.shape) == 2 or img_dep.shape[2] == 1:
        img_dep = cv2.cvtColor(img_dep, cv2.COLOR_GRAY2BGR)

    if img_ir.shape[:2] != (h, w):
        img_ir = cv2.resize(img_ir, (w, h))
    if img_dep.shape[:2] != (h, w):
        img_dep = cv2.resize(img_dep, (w, h))

    variants = []

    for scale in scales:
        img, ratio, pad = letterbox(img_vis, scale)
        img2, _, _ = letterbox(np.concatenate([img_ir, img_dep], axis=2), scale)

        # 原图
        img_t = torch.from_numpy(img.transpose(2, 0, 1)[::-1].copy()).float().unsqueeze(0) / 255.0
        img2_t = torch.from_numpy(img2.transpose(2, 0, 1)[::-1].copy()).float().unsqueeze(0) / 255.0
        variants.append((img_t, img2_t, ratio, pad, False, scale))

        # 水平翻转
        img_f = cv2.flip(img, 1)
        img2_f = cv2.flip(img2, 1)
        img_t_f = torch.from_numpy(img_f.transpose(2, 0, 1)[::-1].copy()).float().unsqueeze(0) / 255.0
        img2_t_f = torch.from_numpy(img2_f.transpose(2, 0, 1)[::-1].copy()).float().unsqueeze(0) / 255.0
        variants.append((img_t_f, img2_t_f, ratio, pad, True, scale))

    return variants, (h, w)


def scale_boxes_norm(boxes, ratio, pad, orig_w, orig_h, flip=False, img_size=1280):
    """反算到原图坐标，再归一化到 0-1（供 WBF 使用）"""
    dw, dh = pad
    if flip:
        boxes[:, [0, 2]] = img_size - boxes[:, [2, 0]]
    boxes[:, [0, 2]] -= dw
    boxes[:, [1, 3]] -= dh
    boxes[:, :4] /= ratio
    boxes[:, [0, 2]] = boxes[:, [0, 2]].clamp(0, orig_w)
    boxes[:, [1, 3]] = boxes[:, [1, 3]].clamp(0, orig_h)
    boxes[:, [0, 2]] /= orig_w
    boxes[:, [1, 3]] /= orig_h
    return boxes


def extract_tensors(obj):
    result = []
    if isinstance(obj, torch.Tensor):
        result.append(obj)
    elif isinstance(obj, (list, tuple)):
        for item in obj:
            result.extend(extract_tensors(item))
    return result


def main():
    out_dir.mkdir(exist_ok=True)
    print(f"加载模型: {weights_path}")
    model = YOLO(weights_path)
    model.model.eval().cuda()

    vis_dir = test_dir / "visible"
    img_files = sorted([f for f in os.listdir(vis_dir)
                        if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    print(f"共 {len(img_files)} 张测试图像")
    print(f"TTA: {len(tta_scales)} 个尺度 × 2 种翻转 = {len(tta_scales)*2} 次推理/图")

    for idx, img_name in enumerate(img_files):
        base_name = Path(img_name).stem
        vis_path = vis_dir / img_name
        ir_path = test_dir / "infrared" / img_name
        dep_path = test_dir / "depth" / img_name

        variants, (orig_h, orig_w) = preprocess_multiscale(vis_path, ir_path, dep_path, tta_scales)

        all_boxes = []
        all_scores = []
        all_labels = []

        for img_t, img2_t, ratio, pad, flip, scale in variants:
            with torch.no_grad():
                pred = model.model(img_t.cuda(), img2_t.cuda())

            all_tensors = extract_tensors(pred)
            det_tensor = None
            for t in all_tensors:
                if t.shape[1] == 16:
                    if t.dim() == 3:
                        det_tensor = t
                        break
                    elif t.dim() == 4:
                        det_tensor = t.flatten(2)
                        break

            if det_tensor is None:
                continue

            det = non_max_suppression(det_tensor, conf_thres=conf_thres,
                                      iou_thres=iou_thres, max_det=max_det)[0]

            if det is not None and len(det):
                det[:, :4] = scale_boxes_norm(det[:, :4], ratio, pad, orig_w, orig_h, flip, scale)
                all_boxes.append(det[:, :4].cpu().numpy())
                all_scores.append(det[:, 4].cpu().numpy())
                all_labels.append(det[:, 5].cpu().numpy().astype(int))

        # ========== WBF 融合 ==========
        if len(all_boxes) > 0:
            boxes, scores, labels = weighted_boxes_fusion(
                all_boxes, all_scores, all_labels,
                weights=[1.0] * len(all_boxes),
                iou_thr=0.55,           # WBF 内部 IoU 阈值，密集场景用 0.55
                skip_box_thr=0.01       # 低于 0.01 的框丢弃
            )

            # 关键：按比赛规则截断 top 100
            if len(boxes) > 100:
                idx_top = np.argsort(scores)[::-1][:100]
                boxes = boxes[idx_top]
                scores = scores[idx_top]
                labels = labels[idx_top]

            # 反归一化回原图绝对坐标
            boxes[:, [0, 2]] *= orig_w
            boxes[:, [1, 3]] *= orig_h
        else:
            boxes, scores, labels = np.array([]), np.array([]), np.array([])

        # 保存
        out_path = out_dir / f"{base_name}.txt"
        with open(out_path, 'w') as f:
            for box, score, label in zip(boxes, scores, labels):
                x1, y1, x2, y2 = box
                w = x2 - x1
                h = y2 - y1
                cx = (x1 + w / 2) / orig_w
                cy = (y1 + h / 2) / orig_h
                w = w / orig_w
                h = h / orig_h
                f.write(f"{int(label)} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f} {score:.6f}\n")

        if (idx + 1) % 50 == 0 or idx == 0:
            print(f"  [{idx+1}/{len(img_files)}] {base_name} | 融合后 {len(boxes)} 个框")

    print(f"\n完成！结果保存在: {out_dir.absolute()}")
    print("全选 txt -> 压缩成 zip -> 提交")


if __name__ == '__main__':
    main()