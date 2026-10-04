# -*- coding: utf-8 -*-
"""双验证：① 屏蔽开关真的改数据吗 ② 注入各环节的实时数值。"""
import os, sys, cv2, torch
sys.path.insert(0, "bisai_v5")
import v5_channels; v5_channels.install()
from ultralytics import YOLO
from v5_model import apply_fusion_v5
from v5_dataset import load_tri, RarePasteDatasetV4
from pathlib import Path

df = Path("/root/autodl-tmp/data_full")

# ① 环境变量验证
RarePasteDatasetV4.TRAIN_IR = df/"ir"; RarePasteDatasetV4.VAL_IR = df/"ir_val"
RarePasteDatasetV4.TRAIN_DEPTH = df/"depth"; RarePasteDatasetV4.VAL_DEPTH = df/"depth_val"
NAMES = ["person","boat","animal","seat","sign","bicycle","car","ball","light","garbage_can","uav","tricycle"]
from ultralytics.cfg import get_cfg
ds = RarePasteDatasetV4(img_path=str(df/"splits/val.txt"),
                        data={"names":{i:n for i,n in enumerate(NAMES)},"channels":9},
                        task="detect", augment=False, imgsz=1024, stride=32, hyp=get_cfg())
img0 = ds[0]["img"]
print("[verify] V5_DROP_IR=%s -> ch3 mean=%.4f (应为0)" % (os.environ.get("V5_DROP_IR","0"), float(img0[3].mean())))
print("[verify] V5_DROP_DEPTH=%s -> ch4 mean=%.4f (应为0)" % (os.environ.get("V5_DROP_DEPTH","0"), float(img0[4].mean())))

# ② 注入实时数值（当前训练的 last.pt）
m = YOLO("runs/detect/runs/train/v5_fusion_b/weights/last.pt")
apply_fusion_v5(m.model)
m.model.cuda().eval()
im, _ = load_tri(df/"images/00000033.jpg", df/"ir_val/00000033.jpg", df/"depth_val/00000033.jpg")
im = cv2.resize(im, (640, 384))
x = torch.from_numpy(im).permute(2,0,1)[None].float().cuda()/255.
with torch.no_grad():
    m.model(x)
print("[verify] reliability 门控:", getattr(m.model.reliability, "_dbg", None))
for i, inj in enumerate(m.model.inject):
    print("[verify] inject.%d:" % i, getattr(inj, "_dbg", None))
