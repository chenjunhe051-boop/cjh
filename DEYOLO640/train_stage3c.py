"""Stage 3c (决定性干预): 有界门控 DEA3 + backbone3 重置 + 640 重训

三连败根因: DEA3 的 depth 门控范围 [0,1] 会摧毁 RGB/IR 特征, 且 backbone3
是黑图时代的垃圾权重。本脚本:
  1. 加载 S2 的 last.pt (RGB/IR/head 完好)
  2. backbone3 (model.20~29) 重置为 yolov8m 映射权重 (干净起点学真实 depth)
  3. DEA3 门控已改为有界 [0.5, 1.0] (depth 最多衰减50%, 永不摧毁)

阶段 3c-1 (20轮): freeze=20 冻 RGB/IR, backbone3+head 学真实 depth
阶段 3c-2 (40轮): freeze=0 全解冻精调
"""
import re
import torch
from pathlib import Path
from ultralytics import YOLO
if __name__ == '__main__':
    print("=" * 70)
    print("Stage 3c-1: load S2 last.pt, reset backbone3, freeze=20 (20 epochs)")
    print("=" * 70)

    s2_path = "my_project/exp_3mod_s2/weights/last.pt"
    if not Path(s2_path).exists():
        s2_path = "my_project/exp_3mod_s2/weights/best.pt"

    model = YOLO(s2_path)

    # ===== 重置 backbone3: yolov8m 的 backbone (model.0~9) → model.20~29 =====
    ckpt = torch.load("yolov8m.pt", map_location='cuda', weights_only=False)
    pretrained_dict = ckpt['model'].state_dict()
    sd = model.model.state_dict()

    reset_cnt = 0
    for k, v in pretrained_dict.items():
        new_k = re.sub(r'^model\.(\d+)', lambda m: f'model.{int(m.group(1)) + 20}', k)
        if new_k in sd and v.shape == sd[new_k].shape:
            sd[new_k] = v
            reset_cnt += 1

    model.model.load_state_dict(sd, strict=False)
    print(f"backbone3 reset: {reset_cnt} layers from yolov8m.pt")
    print(f"Loaded {s2_path} (RGB/IR/head preserved)")

    model.train(
        data="mydata.yaml",
        epochs=20,
        imgsz=640,
        batch=8,
        device="cuda",
        cache=True,
        augment=True,
        workers=0,
        amp=True,
        patience=20,
        project="my_project",
        name="exp_3mod_s3c1",
        save_period=10,
        plots=False,
        rect=False,
        freeze=20,  # 冻结 backbone1(0~9) + backbone2(10~19)

        box=7.5,
        cls=0.3,
        dfl=1.5,
        weight_decay=0.001,
        label_smoothing=0.1,

        mixup=0.1,
        mosaic=1.0,
        close_mosaic=5,
        copy_paste=0.3,
        scale=0.9,
        degrees=5.0,
        translate=0.1,
        shear=2.0,
        flipud=0.5,
        fliplr=0.5,

        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,

        cos_lr=True,
        lr0=0.005,  # backbone3 从干净预训练学真实 depth
        lrf=0.00005,
    )

    # ==================== 阶段 3c-2: 全解冻精调 ====================
    print("\n" + "=" * 70)
    print("Stage 3c-2: freeze=0, full fine-tuning (40 epochs)")
    print("=" * 70)

    s3c1_path = "my_project/exp_3mod_s3c1/weights/last.pt"
    if not Path(s3c1_path).exists():
        s3c1_path = "my_project/exp_3mod_s3c1/weights/best.pt"

    model2 = YOLO(s3c1_path)
    print(f"Loaded {s3c1_path}")

    model2.train(
        data="mydata.yaml",
        epochs=40,
        imgsz=640,
        batch=8,
        device="cuda",
        cache=True,
        augment=True,
        workers=8,
        amp=True,
        patience=20,
        project="my_project",
        name="exp_3mod_s3c2",
        save_period=10,
        plots=False,
        rect=False,
        freeze=0,

        box=7.5,
        cls=0.3,
        dfl=1.5,
        weight_decay=0.001,
        label_smoothing=0.1,

        mixup=0.1,
        mosaic=1.0,
        close_mosaic=10,  # 最后 10 轮真实图精修
        copy_paste=0.3,
        scale=0.9,
        degrees=5.0,
        translate=0.1,
        shear=2.0,
        flipud=0.5,
        fliplr=0.5,

        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,

        cos_lr=True,
        lr0=0.001,
        lrf=0.00001,
    )

    print("\nStage 3c done. Submit with exp_3mod_s3c2/weights/last.pt")

