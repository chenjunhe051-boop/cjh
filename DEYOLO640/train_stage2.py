"""Stage 2: 加载 Stage 1 的 best.pt，全解冻低 lr 精调 (180 epoch)"""
import torch
import re
from pathlib import Path
from ultralytics import YOLO

print("=" * 70)
print("Stage 2: freeze=0, low lr fine-tuning (max 180 epochs)")
print("=" * 70)

# 加载 Stage 1 训练好的权重
model = YOLO("my_project/exp_3mod_s1/weights/best.pt")
print("Loaded Stage 1 best.pt")

model.train(
    data="mydata.yaml",
    epochs=100,
    imgsz=640,
    batch=8,
    device="cuda",
    cache=True,
    augment=True,
    workers=8,
    amp=True,
    patience=20,
    project="my_project",
    name="exp_3mod_s2",
    save_period=15,
    plots=False,
    rect=False,
    freeze=0,                # 全解冻

    box=7.5,
    cls=0.3,
    dfl=1.5,
    weight_decay=0.001,
    label_smoothing=0.1,

    mixup=0.1,
    mosaic=1.0,
    close_mosaic=15,
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
    lr0=0.001,               # 全解冻低 lr
    lrf=0.00001,
)
