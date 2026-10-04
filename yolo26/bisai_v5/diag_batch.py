# -*- coding: utf-8 -*-
"""训练批数据解剖：直接取融合训练同配置下的 3 个样本，检查通道含义和标签。"""
import os, sys
from pathlib import Path

os.chdir("/root/autodl-tmp/YOLO-Master")
sys.path.insert(0, "bisai_v5")

import v5_channels
v5_channels.install()

import ultralytics.data.dataset as ds_mod
import ultralytics.data.build as bd_mod
from v5_dataset import RarePasteDatasetV4

data_full = Path("/root/autodl-tmp/data_full")
RarePasteDatasetV4.TRAIN_IR = data_full / "ir"
RarePasteDatasetV4.VAL_IR = data_full / "ir_val"
RarePasteDatasetV4.TRAIN_DEPTH = data_full / "depth"
RarePasteDatasetV4.VAL_DEPTH = data_full / "depth_val"

NAMES = ["person", "boat", "animal", "seat", "sign", "bicycle",
         "car", "ball", "light", "garbage_can", "uav", "tricycle"]
spec = {"names": {i: n for i, n in enumerate(NAMES)}, "channels": 9}

from ultralytics.cfg import get_cfg
hyp = get_cfg()

ds = RarePasteDatasetV4(
    img_path=str(data_full / "splits" / "train.txt"),
    data=spec, task="detect", augment=True,
    imgsz=1024, stride=32, batch=4, hyp=hyp)
print("dataset size:", len(ds))
print("buffer len:", len(getattr(ds, "buffer", [])))

for i in [0, 5, 100]:
    item = ds[i]
    img = item["img"]
    if hasattr(img, "numpy"):
        img = img.numpy()
    cls = item["cls"]
    bb = item["bboxes"]
    print("=== sample", i, "===")
    print("  img shape:", img.shape, "dtype:", img.dtype,
          "min/max:", float(img.min()), float(img.max()))
    print("  ch means:", [round(float(img[c].mean()), 2) for c in range(img.shape[0])])
    print("  n_bbox:", 0 if bb is None else len(bb),
          "cls:", [] if cls is None else [int(x) for x in cls.flatten().tolist()][:12])
    if bb is not None and len(bb):
        print("  bbox[0]:", [round(float(v), 4) for v in bb[0]])
