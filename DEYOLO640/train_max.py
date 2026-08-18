"""DEYOLO 三模态训练 — RGB + IR + Depth (scale m, 68.5M params)

训练策略:
  - 阶段1 (20 epoch): freeze=10 只冻 backbone1(RGB), 训练 backbone2/3 + head
    让 head 梯度修正 backbone2/3 的 C2f_BiFocus 随机层
  - 阶段2 (180 epoch): freeze=0 全解冻, lr0=0.001 低学习率精调
    真正的分阶段: head 收敛后, backbone 全部参与微调
  - 当前分辨率: 640 (快速验证), 验证通过后切 1280 正式训
"""

import torch
import re
from pathlib import Path
from ultralytics import YOLO


# ========================= 阶段0: 构建模型 + 预训练权重映射 =========================
print("=" * 70)
print("Stage 0: Build model & load pretrained weights")
print("=" * 70)

model = YOLO("ultralytics/models/v8/DEYOLO.yaml")

ckpt = torch.load("yolov8m.pt", map_location='cuda', weights_only=False)
pretrained_dict = ckpt['model'].state_dict()
model_dict = model.model.state_dict()

matched_dict = {}
skipped = []

for k, v in pretrained_dict.items():
    # 每个 pretrained key 同时映射到 3 个 backbone: 直接(0) + offset 10 + offset 20
    # 例: model.0.conv.weight → model.0/10/20.conv.weight
    mapped = False
    for offset in [0, 10, 20]:
        if offset == 0:
            new_k = k
        else:
            new_k = re.sub(r'^model\.(\d+)', lambda m: f'model.{int(m.group(1)) + offset}', k)
        if new_k in model_dict and v.shape == model_dict[new_k].shape:
            matched_dict[new_k] = v
            mapped = True
    if not mapped:
        skipped.append(k)

model.model.load_state_dict(matched_dict, strict=False)

print(f"Loaded {len(matched_dict)}/{len(model_dict)} layers from yolov8m.pt")
print(f"  - backbone1 (model.0~9) : direct match")
print(f"  - backbone2 (model.10~19): mapped from backbone1")
print(f"  - backbone3 (model.20~29): mapped from backbone1")
print(f"Skipped {len(skipped)} layers (DEA3 + head → random init, expected)")


# ========================= 阶段1: 冻 backbone1, 训 backbone2/3 + head (20 epoch) =========================
print("\n" + "=" * 70)
print("Stage 1: freeze=10, train backbone2/3 + head (20 epochs)")
print("=" * 70)

model.train(
    data="mydata.yaml",
    epochs=20,
    imgsz=640,
    batch=8,
    device="cuda",
    cache=True,
    augment=True,
    workers=8,
    amp=True,
    patience=20,
    project="my_project",
    name="exp_3mod_s1",
    save_period=5,
    plots=False,
    rect=False,
    freeze=10,               # 只冻结 backbone1(RGB锚点), backbone2/3 可训练修正 C2f_BiFocus

    # 损失权重: mAP50-95 需要定位+分类平衡，cls 过高会导致高 IoU 阈值下分类错配
    box=7.5,
    cls=0.3,
    dfl=1.5,

    # 正则化: 2000 张小数据集，weight_decay 和 label_smoothing 稍微给足
    weight_decay=0.001,
    label_smoothing=0.1,

    # 数据增强
    mixup=0.1,               # 三模态 mixup 对定位精度有损，保持 0.1 别太高
    mosaic=1.0,
    close_mosaic=5,          # 阶段1只有 20 轮，最后 5 轮关 mosaic 做稳定收敛
    copy_paste=0.3,          # 小目标(ball/uav/seat/light)救命增强
    scale=0.9,               # 10%~190% 随机缩放，小目标场景必须
    degrees=5.0,
    translate=0.1,
    shear=2.0,
    flipud=0.5,
    fliplr=0.5,

    # 颜色增强: 只对 RGB 有效(IR/Depth 在 augment 里已跳过 HSV)
    hsv_h=0.015,
    hsv_s=0.7,
    hsv_v=0.4,

    # 学习率: backbone2/3 可训练，lr 保守防破坏预训练权重
    cos_lr=True,
    lr0=0.005,
    lrf=0.00005,
)


