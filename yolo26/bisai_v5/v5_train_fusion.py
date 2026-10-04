# -*- coding: utf-8 -*-
"""v5_train_fusion -- 9 通道三模态融合训练（自包含，不依赖 bisai_v4）。

两段式（对应朋友方案的 Phase A / Phase B）：
  Phase A（--freeze-backbone）：RGB 骨干冻结，只训融合支路 + 检测头；
      inject warmup 从 0 线性升温 --warmup-epochs 个 epoch。
      warmup=0 时融合模型逐值等于 RGB 种子，所以全程不应低于种子分数。
  Phase B（--init-from PhaseA/best.pt，不带 --freeze-backbone）：全模型低学习率微调。

用法（在 /root/autodl-tmp/YOLO-Master 下运行， ultralytics 目录的上一级）：
  # Phase A
  python bisai_v5/v5_train_fusion.py \
      --seed runs/detect/runs/train/y26x_o365/weights/best.pt \
      --data /root/autodl-tmp/data_full \
      --imgsz 1024 --epochs 60 --batch 4 --lr 1e-3 \
      --freeze-backbone --warmup-epochs 10 --name v5_fusion_a

  # Phase B
  python bisai_v5/v5_train_fusion.py \
      --seed runs/detect/runs/train/y26x_o365/weights/best.pt \
      --init-from runs/train/v5_fusion_a/weights/best.pt \
      --data /root/autodl-tmp/data_full \
      --imgsz 1024 --epochs 40 --batch 4 --lr 1e-4 --name v5_fusion_b
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

NAMES = ["person", "boat", "animal", "seat", "sign", "bicycle",
         "car", "ball", "light", "garbage_can", "uav", "tricycle"]


def find_data(explicit=""):
    cands = [Path(explicit)] if explicit else []
    cands += [Path('/root/autodl-tmp/data_full'), Path('/root/autodl-tmp/dataset'),
              Path('/root/autodl-tmp/mydata'), Path('data_full')]
    for c in cands:
        if c.is_dir() and (c / 'images').is_dir() and (c / 'ir').is_dir() \
                and (c / 'depth').is_dir() and (c / 'splits').is_dir():
            return c.resolve()
    return None


def write_yaml(data, yaml_path):
    names_yaml = "\n".join("  %d: %s" % (i, n) for i, n in enumerate(NAMES))
    Path(yaml_path).write_text(
        "path: %s\ntrain: splits/train.txt\nval: splits/val.txt\nchannels: 9\nnames:\n%s\n"
        % (data, names_yaml), encoding='utf-8')
    return str(yaml_path)


def main():
    ap = argparse.ArgumentParser(description="v5 9通道三模态融合训练（自包含）")
    ap.add_argument("--seed", required=True, help="RGB 种子权重（你的 y26x_o365 best.pt）")
    ap.add_argument("--init-from", default=None,
                    help="已训练过的 v5 融合权重（Phase B 用它恢复融合支路）")
    ap.add_argument("--data", default=os.environ.get("DATA_FULL", ""))
    ap.add_argument("--imgsz", type=int, default=1024)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--no-resume", action="store_true", help="忽略断点强制从头训练")
    ap.add_argument("--optimizer", default="AdamW", choices=["AdamW", "MuSGD"],
                    help="MuSGD lr 建议 0.01 量级（fork 原生分组学习率）")
    ap.add_argument("--lrf", type=float, default=0.01)
    ap.add_argument("--patience", type=int, default=25)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--device", default="0")
    ap.add_argument("--project", default="runs/train")
    ap.add_argument("--name", default="v5_fusion_a")
    ap.add_argument("--freeze-backbone", action="store_true",
                    help="Phase A：冻结 RGB 骨干，只训融合支路 + 检测头（保证不跌种子）")
    ap.add_argument("--warmup-delay", type=int, default=0)
    ap.add_argument("--warmup-epochs", type=int, default=10,
                    help="inject warmup 从 0 线性升到 1 所需 epoch 数")
    ap.add_argument("--close-mosaic", type=int, default=10)
    ap.add_argument("--modal-drop", type=float, default=0.25,
                    help="模态随机丢弃概率（鲁棒性训练）")
    ap.add_argument("--rare-paste", type=float, default=0.3,
                    help="稀有类(ball/tricycle) copy-paste 概率")
    ap.add_argument("--no-amp", action="store_true")
    ap.add_argument("--wio", action="store_true",
                    help="用 WIoU v3 替换 CIoU（提升 mAP50-95 精细定位，论文 05）")
    ap.add_argument("--scale-gain", type=float, default=0.5,
                    help="缩放抖动强度（论文 04 LSJ 建议 0.9；小目标收益大）")
    ap.add_argument("--seed-int", type=int, default=0)
    args = ap.parse_args()

    import v5_channels
    v5_channels.install()
    if args.wio or os.environ.get("V5_WIOU", "0") == "1":
        from v5_wiou import install_wiou
        install_wiou()

    data = find_data(args.data)
    if data is None:
        sys.exit("[error] 找不到数据目录（需要 images/ ir/ depth/ splits/ 四个子目录），"
                 "用 --data 指定，或先做软链接（见 README.md）")
    print("[train] data:", data)

    from v5_dataset import RarePasteDatasetV4
    RarePasteDatasetV4.TRAIN_IR = data / "ir"
    RarePasteDatasetV4.TRAIN_DEPTH = data / "depth"
    RarePasteDatasetV4.VAL_IR = [d for d in (data / "ir_val", data / "ir") if d.is_dir()]
    RarePasteDatasetV4.VAL_DEPTH = [d for d in (data / "depth_val", data / "depth") if d.is_dir()]
    RarePasteDatasetV4.PASTE_PROB = args.rare_paste
    os.environ["V5_MODAL_DROP"] = str(args.modal_drop)

    yaml_path = write_yaml(data, HERE / "v5_fusion_data.yaml")

    # 让 trainer 的 train / val dataloader 都走 9 通道数据集
    import ultralytics.data.build as bd_mod
    import ultralytics.data.dataset as ds_mod
    ds_mod.YOLODataset = RarePasteDatasetV4
    bd_mod.YOLODataset = RarePasteDatasetV4
    print("[train] train/val dataset -> RarePasteDatasetV4 (9ch)")

    from ultralytics import YOLO
    from v5_model import apply_fusion_v5, set_inject_warmup

    model = YOLO(args.seed)
    aux_src = None
    if args.init_from:
        ck = torch.load(args.init_from, map_location="cpu", weights_only=False)
        aux_src = ck.get("model", None) if isinstance(ck, dict) else None
        print("[init] 从 %s 恢复融合支路权重" % args.init_from)
    apply_fusion_v5(model.model, aux_src)
    set_inject_warmup(model.model, 0.0)

    if args.freeze_backbone:
        n_layers = len(model.model.model)
        freeze = list(range(max(0, n_layers - 1)))   # 冻骨干，留检测头可训
        print("[freeze] 冻结 RGB 骨干 %d 层（检测头 + 融合支路可训）" % len(freeze))
    else:
        freeze = []

    def _on_epoch_start(trainer):
        w = min(1.0, max(0.0, (trainer.epoch - args.warmup_delay)
                          / max(1, args.warmup_epochs)))
        set_inject_warmup(trainer.model, w)
        ema = getattr(trainer, "ema", None)   # EMA 是深拷贝：不同步则保存的 ckpt warmup 恒为 0
        if ema is not None and getattr(ema, "ema", None) is not None:
            set_inject_warmup(ema.ema, w)
        if trainer.epoch < 3 or trainer.epoch == args.warmup_epochs:
            print("[warmup] epoch %d -> inject warmup=%.3f" % (trainer.epoch, w))

    model.add_callback("on_train_epoch_start", _on_epoch_start)

    kw = dict(
        model=str(Path(args.seed).resolve()),
        data=str(yaml_path), imgsz=args.imgsz, epochs=args.epochs,
        batch=args.batch, optimizer=args.optimizer, lr0=args.lr, lrf=args.lrf,
        weight_decay=0.005, cos_lr=True, warmup_epochs=3,
        patience=args.patience, box=7.5, dfl=1.5,
        close_mosaic=args.close_mosaic, rect=False, val=True, save=True,
        workers=args.workers, device=args.device, verbose=True,
        mosaic=0.6, mixup=0.15, copy_paste=0.0,
        degrees=5.0, translate=0.15, scale=args.scale_gain, fliplr=0.5, flipud=0.0,
        hsv_h=0.08, hsv_s=0.75, hsv_v=0.60,
        erasing=0.25, auto_augment=None,
        label_smoothing=0.05, freeze=freeze, seed=args.seed_int,
        plots=False, project=args.project, name=args.name, exist_ok=True,
        amp=not args.no_amp,
    )
    print("[train] imgsz=%d epochs=%d batch=%d lr=%s freeze=%s warmup=%d"
          % (args.imgsz, args.epochs, args.batch, args.lr,
             bool(freeze), args.warmup_epochs))
    # 关键：本 fork 的 get_model 会按 data["channels"] 重建模型 —— channels=9 时
    # 3通道stem权重被丢弃、融合模块整个丢失（epoch1 loss 4.7 / val 0.03 的根因）。
    # 直接把内存中的融合模型交给 trainer（setup_model 检测到 nn.Module 就跳过重建）。
    last_cands = sorted(Path("runs").glob("**/train/%s/weights/last.pt" % args.name))
    if last_cands and last_cands[-1].is_file() and not args.no_resume:
        print("[resume] 检测到断点 %s，从中继续（优化器/轮数/调度器完整恢复）"
              % last_cands[-1])
        m2 = YOLO(str(last_cands[-1]))
        # 修复 resume 的优化器分组不匹配：fork 的 resume 路径在构建优化器之后才应用
        # freeze，重建的优化器把冻结骨干也算进权重组（组变大），与存档对不上。
        # 这里提前冻结，保证 build_optimizer 看到正确的可训参数集。
        if freeze:
            n_layers = len(m2.model.model)
            for i, layer in enumerate(m2.model.model):
                if i >= n_layers - 1:
                    continue
                for p in layer.parameters():
                    p.requires_grad_(False)
        m2.add_callback("on_train_epoch_start", _on_epoch_start)
        m2.train(resume=str(last_cands[-1]))   # 显式传路径，绕开 get_latest_run 的猜测
        t2 = m2.trainer
        sd2 = Path(getattr(t2, "save_dir", Path("runs/detect/runs/train") / args.name))
        b2 = Path(getattr(t2, "best", sd2 / "weights" / "best.pt"))
        print("[resume] 完成，best=%s" % b2)
        return

    from ultralytics.models.yolo.detect import DetectionTrainer
    trainer = DetectionTrainer(overrides=kw, _callbacks=model.callbacks)
    trainer.model = model.model
    print("[train] 已注入内存融合模型（跳过 get_model 重建，stem/aux 完整保留）")
    trainer.train()

    t = trainer
    save_dir = Path(getattr(t, "save_dir", Path(args.project) / args.name))
    best = Path(getattr(t, "best", save_dir / "weights" / "best.pt"))
    print("[done] best = %s" % best)
    try:
        print("[done] final: %s" % {k: round(float(v), 5)
                                    for k, v in t.metrics.items()
                                    if isinstance(v, (int, float))})
    except Exception as e:                                  # noqa: BLE001
        print("[done] metrics 打印失败: %r" % e)
    print("[next] 对照验证：python bisai_v5/v5_initeval.py --seed %s --model %s --imgsz %d"
          % (args.seed, best, args.imgsz))


if __name__ == "__main__":
    main()
