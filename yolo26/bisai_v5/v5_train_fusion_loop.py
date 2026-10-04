# -*- coding: utf-8 -*-
"""v5_train_fusion_loop -- 独立融合训练循环（绕开 fork 训练器）。

动机：本 fork 的训练器在"融合模块 + 9通道 + freeze"组合下表现异常
（epoch1 loss 4.7/val 0.03，而隔离探针全部正常）。本脚本只复用
已验证的部件：RarePasteDatasetV4（数据）、双头损失（与 fork 一致）、
我们的 dispatcher（地板锁）。优化器/调度/EMA/AMP 全部手写、行为透明。

地板锁不变：warmup=0 时模型严格等于种子。
"""
import argparse
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

NAMES = ["person", "boat", "animal", "seat", "sign", "bicycle",
         "car", "ball", "light", "garbage_can", "uav", "tricycle"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", required=True)
    ap.add_argument("--data", default="/root/autodl-tmp/data_full")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--warmup-epochs", type=int, default=15)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--name", default="v5_fusion_loop")
    ap.add_argument("--inject-bias", type=float, default=-2.0)
    ap.add_argument("--save-every", type=int, default=5, help="每 N 轮验证+保存")
    args = ap.parse_args()

    os.environ["V5_INJECT_BIAS"] = str(args.inject_bias)

    import v5_channels
    v5_channels.install()

    import ultralytics.data.dataset as ds_mod
    import ultralytics.data.build as bd_mod
    from v5_dataset import RarePasteDatasetV4

    data = Path(args.data)
    RarePasteDatasetV4.TRAIN_IR = data / "ir"
    RarePasteDatasetV4.TRAIN_DEPTH = data / "depth"
    RarePasteDatasetV4.VAL_IR = data / "ir_val" if (data / "ir_val").is_dir() else data / "ir"
    RarePasteDatasetV4.VAL_DEPTH = data / "depth_val" if (data / "depth_val").is_dir() else data / "depth"
    os.environ["V5_MODAL_DROP"] = "0.25"

    spec = {"names": {i: n for i, n in enumerate(NAMES)}, "channels": 9}
    from ultralytics.cfg import get_cfg
    hyp = get_cfg()
    hyp.mosaic, hyp.mixup, hyp.copy_paste = 0.6, 0.15, 0.0
    hyp.degrees, hyp.translate, hyp.scale, hyp.fliplr = 5.0, 0.15, 0.5, 0.5
    hyp.hsv_h, hyp.hsv_s, hyp.hsv_v = 0.08, 0.75, 0.60
    hyp.close_mosaic = 10
    hyp.auto_augment = None
    hyp.erasing = 0.0
    hyp.label_smoothing = 0.05
    hyp.box, hyp.cls, hyp.dfl = 7.5, 0.5, 1.5

    train_ds = RarePasteDatasetV4(img_path=str(data / "splits" / "train.txt"),
                                  data=spec, task="detect", augment=True,
                                  imgsz=1024, stride=32, hyp=hyp)
    g = torch.Generator(); g.manual_seed(0)
    train_ld = torch.utils.data.DataLoader(
        train_ds, batch_size=args.batch, shuffle=True, num_workers=args.workers,
        collate_fn=RarePasteDatasetV4.collate_fn, pin_memory=True,
        persistent_workers=True, generator=g)

    from ultralytics import YOLO
    from v5_model import apply_fusion_v5, set_inject_warmup

    model = YOLO(args.seed)
    apply_fusion_v5(model.model)
    set_inject_warmup(model.model, 0.0)
    det = model.model
    det.train()

    # 冻结骨干（除检测头外的 model.N 层）
    n_layers = len(det.model)
    frozen = 0
    for i, layer in enumerate(det.model):
        if i >= n_layers - 1:
            continue
        for p in layer.parameters():
            p.requires_grad_(False)
            frozen += 1
    print("[loop] frozen backbone param tensors:", frozen)

    crit = model._smart_load("loss")(model=det)   # 与 fork 训练器同一个损失类

    trainable = [p for p in det.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=0.005)
    nb = len(train_ld)
    total_steps = args.epochs * nb
    warmup_steps = 3 * nb
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / warmup_steps if s < warmup_steps
        else 0.01 + 0.99 * 0.5 * (1 + math.cos(math.pi * (s - warmup_steps) / max(1, total_steps - warmup_steps))))

    # EMA
    ema = {k: v.detach().clone().float() for k, v in det.state_dict().items()}

    save_dir = Path("runs/detect/runs/train") / args.name
    (save_dir / "weights").mkdir(parents=True, exist_ok=True)
    yaml_path = HERE / "v5_fusion_data.yaml"
    names_yaml = "\n".join("  %d: %s" % (i, n) for i, n in enumerate(NAMES))
    yaml_path.write_text(
        "path: %s\ntrain: splits/train.txt\nval: splits/val.txt\nchannels: 9\nnames:\n%s\n"
        % (data, names_yaml), encoding="utf-8")
    ds_mod.YOLODataset = RarePasteDatasetV4   # 验证也走 9 通道
    bd_mod.YOLODataset = RarePasteDatasetV4

    device = "cuda:0"
    det.to(device)
    amp = torch.amp.autocast("cuda", enabled=True)
    scaler = torch.amp.GradScaler("cuda", enabled=True)

    best_map = -1.0
    for epoch in range(args.epochs):
        w = min(1.0, max(0.0, epoch / max(1, args.warmup_epochs)))
        set_inject_warmup(det, w)
        if epoch < 3 or epoch == args.warmup_epochs:
            print("[warmup] epoch %d -> inject warmup=%.3f, lr=%.2e" % (epoch, w, opt.param_groups[0]["lr"]))

        det.train()
        t0 = time.time()
        run_box = run_cls = run_dfl = 0.0
        seen = 0
        for it, batch in enumerate(train_ld):
            img = batch["img"].to(device, non_blocking=True).float() / 255.0
            for k in ("cls", "bboxes", "batch_idx"):
                batch[k] = batch[k].to(device, non_blocking=True)

            with amp:
                pred = det(img)
                loss, parts = crit(pred, batch)
            if not torch.isfinite(loss):
                print("[skip] non-finite loss at epoch %d iter %d" % (epoch, it))
                scaler.update()
                continue
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(trainable, 10.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            with torch.no_grad():
                d = 0.999
                msd = det.state_dict()
                for k, v in ema.items():
                    mv = msd[k]
                    if mv.dtype.is_floating_point:
                        v.mul_(d).add_(mv.detach().float(), alpha=1 - d)
                    else:
                        v.copy_(mv)
            if torch.is_tensor(parts) and parts.numel() >= 3:
                run_box += float(parts.flatten()[0]); run_cls += float(parts.flatten()[1]); run_dfl += float(parts.flatten()[2])
            seen += 1
        dt = time.time() - t0
        print("epoch %2d/%d  box %.3f cls %.3f dfl %.4f  (%.0fs, lr %.2e)"
              % (epoch + 1, args.epochs, run_box / max(1, seen), run_cls / max(1, seen),
                 run_dfl / max(1, seen), dt, opt.param_groups[0]["lr"]), flush=True)

        if (epoch + 1) % args.save_every == 0 or epoch == args.epochs - 1:
            ck = {k: v for k, v in ema.items()}
            det.load_state_dict(ck, strict=True)
            torch.save({"model": det, "epoch": epoch}, save_dir / "weights" / "last.pt")
            val_map, val_map50 = -1.0, -1.0
            try:
                det.eval()
                metrics = model.val(data=str(yaml_path), imgsz=1024, batch=4,
                                    split="val", device=device, verbose=False, plots=False)
                val_map, val_map50 = float(metrics.box.map), float(metrics.box.map50)
                print("  -> val mAP50-95=%.5f mAP50=%.5f" % (val_map, val_map50), flush=True)
            except Exception as e:
                print("  -> val 失败: %r" % e)
            if val_map > best_map:
                best_map = val_map
                torch.save({"model": det, "epoch": epoch, "val_map": val_map},
                           save_dir / "weights" / "best.pt")
                print("  -> new best (%.5f), saved" % val_map, flush=True)

    print("[loop] done. best val mAP50-95=%.5f" % best_map)


if __name__ == "__main__":
    main()
