# -*- coding: utf-8 -*-
"""训练链路探针：数据加载器 -> 预处理 -> 前向 -> 损失，逐步打印体检值。"""
import os, sys
os.chdir("/root/autodl-tmp/YOLO-Master")
sys.path.insert(0, "bisai_v5")

import v5_channels
v5_channels.install()

import torch
from pathlib import Path

import ultralytics.data.dataset as ds_mod
import ultralytics.data.build as bd_mod
from v5_dataset import RarePasteDatasetV4

df = Path("/root/autodl-tmp/data_full")
RarePasteDatasetV4.TRAIN_IR = df / "ir"
RarePasteDatasetV4.VAL_IR = df / "ir_val"
RarePasteDatasetV4.TRAIN_DEPTH = df / "depth"
RarePasteDatasetV4.VAL_DEPTH = df / "depth_val"

NAMES = ["person", "boat", "animal", "seat", "sign", "bicycle",
         "car", "ball", "light", "garbage_can", "uav", "tricycle"]
spec = {"names": {i: n for i, n in enumerate(NAMES)}, "channels": 9}

from ultralytics.cfg import get_cfg
hyp = get_cfg()

ds = RarePasteDatasetV4(img_path=str(df / "splits" / "train.txt"), data=spec,
                        task="detect", augment=True, imgsz=1024, stride=32, hyp=hyp)
loader = torch.utils.data.DataLoader(
    ds, batch_size=4, shuffle=True, num_workers=0,
    collate_fn=RarePasteDatasetV4.collate_fn, pin_memory=True)

from ultralytics import YOLO
from v5_model import apply_fusion_v5, set_inject_warmup

m = YOLO("runs/detect/runs/train/rgbseed1024_s3/weights/best.pt")
apply_fusion_v5(m.model)
set_inject_warmup(m.model, 0.0)
m.model.train()

crit = m._smart_load("loss")(model=m.model)   # 与 fork 训练器完全同一个损失类

print("=" * 60)
for i, batch in enumerate(loader):
    img = batch["img"].float() / 255.0          # 与 trainer.preprocess_batch 相同
    print("--- batch", i, "---")
    print("  img ch means:", [round(float(img[0, c].mean()), 3) for c in range(9)],
          " max:", round(float(img.max()), 2))
    print("  targets:", tuple(batch["cls"].shape),
          " cls[:8]:", [int(x) for x in batch["cls"][:8].flatten().tolist()])
    with torch.no_grad():
        pred = m.model(img)
    if isinstance(pred, dict):
        print("  pred is DICT, keys:", list(pred.keys()))
        for k, v in pred.items():
            if torch.is_tensor(v):
                print("    %-18s %-22s mean %.4f std %.4f" % (k, tuple(v.shape), float(v.mean()), float(v.std())))
            elif isinstance(v, (list, tuple)):
                print("    %-18s list[%d], [0]:%s" % (k, len(v), tuple(v[0].shape) if torch.is_tensor(v[0]) else type(v[0])))
    else:
        print("  pred:", tuple(pred.shape),
              " mean %.4f std %.4f absmax %.2f" % (float(pred.mean()), float(pred.std()), float(pred.abs().max())))
    for name, feats in pred.items():
        if isinstance(feats, (list, tuple)) and torch.is_tensor(feats[0]):
            for j, f in enumerate(feats):
                print("    %s[%d] %s mean %.4f std %.4f absmax %.1f"
                      % (name, j, tuple(f.shape), float(f.mean()), float(f.std()), float(f.abs().max())))
    try:
        loss, parts = crit(pred, batch)   # 与 trainer 内部调用方式完全一致
        if torch.is_tensor(parts):
            pv = [float(x) for x in parts.flatten().tolist()]
        elif isinstance(parts, dict):
            pv = [float(v) for v in parts.values() if torch.is_tensor(v)]
        else:
            pv = []
        print("  LOSS total=%.4f  items=%s" % (float(loss), [round(x, 4) for x in pv][:6]))
    except Exception as e:
        import traceback; traceback.print_exc()
        print("  LOSS 计算失败: %r" % e)
    if i >= 2:
        break
print("=" * 60)
print("对照：种子训练 epoch1 的 cls 约 0.58；若此处 cls 也是 ~0.5，")
print("说明数据/前向/损失全链路正常，问题在训练器内部（优化器/EMA/AMP）。")
