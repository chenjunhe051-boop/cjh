"""exp2 (44分模型) 测试集预测 — 无TTA单尺度版

exp2_640aug 训练于 640, 预测也用 640。
输出: predictions_exp2/ 每图一个 txt (class cx cy w h conf)
"""
import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))
import cv2, numpy as np, torch
from pathlib import Path
from ultralytics import YOLO
from ultralytics.yolo.utils.ops import non_max_suppression

weights_path = "my_project/exp2_640aug/weights/best.pt"
test_dir = Path("test_data")
out_dir = Path("predictions_exp2")
imgsz = 640          # exp2_640aug 的训练分辨率
conf_thres = 0.05
iou_thres = 0.45
max_det = 100


def letterbox(im, new_shape=640, color=114):
    shape = im.shape[:2]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
    dw /= 2; dh /= 2
    if shape[::-1] != new_unpad:
        c = im.shape[2]
        if c <= 4:
            im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
        else:
            parts = [cv2.resize(im[:, :, i:i+4], new_unpad, interpolation=cv2.INTER_LINEAR)
                     for i in range(0, c, 4)]
            im = np.concatenate(parts, axis=2)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = np.pad(im, ((top, bottom), (left, right), (0, 0)),
                mode='constant', constant_values=color)
    return im, r, (dw, dh)


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
    print(f"共 {len(img_files)} 张测试图像 | imgsz={imgsz} | 无TTA")

    for idx, img_name in enumerate(img_files):
        base_name = Path(img_name).stem
        h, w = cv2.imread(str(vis_dir / img_name)).shape[:2]

        img_vis = cv2.imread(str(vis_dir / img_name))
        img_ir = cv2.imread(str(test_dir / "infrared" / img_name))
        img_dep = cv2.imread(str(test_dir / "depth" / img_name), cv2.IMREAD_UNCHANGED)

        if img_vis is None:
            raise FileNotFoundError(f"无法读取: {img_name}")
        if img_ir is None:
            img_ir = np.zeros((h, w, 3), dtype=np.uint8)
        if img_dep is None:
            img_dep = np.zeros((h, w), dtype=np.uint16)

        # 深度三通道编码 (与训练一致)
        img_dep = np.clip(img_dep, 0, 20000).astype(np.float32)
        if len(img_dep.shape) == 3:
            img_dep = img_dep[:, :, 0]
        depth_norm = img_dep / 20000.0
        depth_log = (np.log1p(depth_norm * 10) / np.log1p(10) * 255).astype(np.uint8)
        depth_linear = (depth_norm * 255).astype(np.uint8)
        depth_grad = cv2.Sobel(depth_linear, cv2.CV_8U, 1, 1)
        if len(depth_grad.shape) == 2:
            depth_grad = depth_grad[..., None]
        if len(depth_log.shape) == 2:
            depth_log = depth_log[..., None]
        if len(depth_linear.shape) == 2:
            depth_linear = depth_linear[..., None]
        img_dep = np.concatenate([depth_log, depth_linear, depth_grad], axis=2)

        if img_ir.shape[:2] != (h, w):
            img_ir = cv2.resize(img_ir, (w, h))
        if img_dep.shape[:2] != (h, w):
            img_dep = cv2.resize(img_dep, (w, h))

        img, ratio, pad = letterbox(img_vis, imgsz)
        img2, _, _ = letterbox(np.concatenate([img_ir, img_dep], axis=2), imgsz)

        im_t = torch.from_numpy(img.transpose(2, 0, 1)[::-1].copy()).float().unsqueeze(0) / 255.0
        im2_t = torch.from_numpy(img2.transpose(2, 0, 1)[::-1].copy()).float().unsqueeze(0) / 255.0

        with torch.no_grad():
            pred = model.model(im_t.cuda(), im2_t.cuda())

        det_tensor = None
        for t in extract_tensors(pred):
            if t.shape[1] == 16:
                det_tensor = t if t.dim() == 3 else t.flatten(2)
                break

        det = non_max_suppression(det_tensor, conf_thres=conf_thres,
                                  iou_thres=iou_thres, max_det=max_det)[0]

        lines = []
        if det is not None and len(det):
            dw, dh = pad
            det[:, [0, 2]] -= dw
            det[:, [1, 3]] -= dh
            det[:, :4] /= ratio
            det[:, [0, 2]] = det[:, [0, 2]].clamp(0, w)
            det[:, [1, 3]] = det[:, [1, 3]].clamp(0, h)

            for d in det:
                x1, y1, x2, y2, conf, cls = d.tolist()
                bw, bh = x2 - x1, y2 - y1
                if bw <= 0 or bh <= 0:
                    continue
                cx, cy = (x1 + x2) / 2 / w, (y1 + y2) / 2 / h
                lines.append(f"{int(cls)} {cx:.6f} {cy:.6f} {bw/w:.6f} {bh/h:.6f} {conf:.6f}")

        (out_dir / f"{base_name}.txt").write_text("\n".join(lines) + ("\n" if lines else ""))

        if (idx + 1) % 100 == 0 or idx == 0:
            print(f"  [{idx+1}/{len(img_files)}] {base_name} | {len(lines)} 个框")

    print(f"\n完成! 结果保存在: {out_dir.absolute()}")


if __name__ == "__main__":
    main()
