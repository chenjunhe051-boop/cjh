# -*- coding: utf-8 -*-
"""v5_train_rgbseed -- 纯 RGB 种子训练（完整复刻朋友路线，自包含）。

为什么需要它：冲 55 要复刻朋友的完整配方 —— 先练一个干净的纯 RGB 种子
（3 通道 stem），再用 v5_train_fusion.py 把融合叠上去（此时 legacy 模式
自动关闭，地板 = RGB 种子分数）。

在你这台被魔改过的 YOLO-Master 上，v5_channels.install() 会自动中和
base.py 的"强制9通道"补丁（仅本进程），让 3 通道训练正常进行。

三段式（与朋友的 seed 训练器一致）：s1=640 / s2=640 / s3=728，
段间用 best.pt 接力；每段完成后写 PHASE_DONE，重跑自动跳过已完成段。

用法（在 YOLO-Master 目录下，tmux 里跑）：
  python bisai_v5/v5_train_rgbseed.py \
      --model yolo26x-objv1-150.pt \
      --data /root/autodl-tmp/data_full \
      --name rgbseed
  # 时间紧张可用 --epochs 60,25,25 缩短（朋友默认 80,40,40）
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
        if c.is_dir() and (c / 'images').is_dir() and (c / 'splits').is_dir():
            return c.resolve()
    return None


def write_yaml(data, yaml_path):
    names_yaml = "\n".join("  %d: %s" % (i, n) for i, n in enumerate(NAMES))
    Path(yaml_path).write_text(
        "path: %s\ntrain: splits/train.txt\nval: splits/val.txt\nchannels: 3\nnames:\n%s\n"
        % (data, names_yaml), encoding='utf-8')
    return str(yaml_path)


def parse_list(s, cast, n=3):
    vals = [cast(x.strip()) for x in str(s).split(",") if x.strip()]
    if len(vals) == 1:
        return vals * n
    if len(vals) != n:
        sys.exit("[error] 需要 1 个或 %d 个逗号分隔的值，收到: %s" % (n, s))
    return vals


def resolve_weights(name):
    """本地权重优先（cwd / YOLO-Master 根 / 常见缓存目录），否则交给 ultralytics。"""
    roots = [Path.cwd(), Path.cwd().parent, Path("/root/autodl-tmp"),
             Path("/root/autodl-tmp/YOLO-Master"),
             Path.home() / ".config" / "Ultralytics",
             Path.home() / ".cache" / "ultralytics"]
    for r in roots:
        for p in (r / name, r / "weights" / name):
            if p.is_file():
                print("[weights] 用本地权重 %s" % p.resolve())
                return str(p.resolve())
    if "/" in name or "\\" in name:
        sys.exit("[error] 权重文件不存在: %s" % name)
    print("[weights] 本地没有找到 %s，交给 ultralytics 解析/下载（需联网）" % name)
    return name


def main():
    ap = argparse.ArgumentParser(description="纯RGB种子训练（复刻朋友配方）")
    ap.add_argument("--model", default="yolo26x-objv1-150.pt",
                    help="O365 预训练快照（默认朋友的 yolo26x-objv1-150.pt）")
    ap.add_argument("--data", default=os.environ.get("DATA_FULL", ""))
    ap.add_argument("--name", default="rgbseed")
    ap.add_argument("--project", default="runs/train")
    ap.add_argument("--imgsz", default="640,640,728")
    ap.add_argument("--epochs", default="80,40,40", help="朋友默认；赶时间可 60,25,25")
    ap.add_argument("--lr", default="3e-4,1e-4,3e-5")
    ap.add_argument("--batch", default="4,4,4", help="显存不够改成 2,2,2")
    ap.add_argument("--optimizer", default="AdamW", choices=["AdamW", "MuSGD"],
                    help="朋友55分用的MuSGD（fork原生，分组学习率）；默认AdamW")
    ap.add_argument("--patience", type=int, default=40)
    ap.add_argument("--close-mosaic", type=int, default=10)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--device", default="0")
    ap.add_argument("--seed-int", type=int, default=0)
    args = ap.parse_args()

    import v5_channels
    v5_channels.install()          # 关键：中和 base.py 的强制9通道补丁

    data = find_data(args.data)
    if data is None:
        sys.exit("[error] 找不到数据目录（需要 images/ 和 splits/），用 --data 指定")
    print("[seed] data:", data)
    yaml_path = write_yaml(data, HERE / "v5_rgb_data.yaml")

    from ultralytics import YOLO

    imsz = parse_list(args.imgsz, int)
    epochs = parse_list(args.epochs, int)
    lrs = parse_list(args.lr, float)
    batches = parse_list(args.batch, int)

    prev = resolve_weights(args.model)
    for i in range(3):
        tag = "%s_s%d" % (args.name, i + 1)
        out_dir = Path(args.project) / tag
        done = out_dir / "PHASE_DONE"
        done_marks = [done] + sorted(Path("runs").glob("**/train/%s/PHASE_DONE" % tag))
        if any(d.is_file() for d in done_marks):
            cands = sorted(Path("runs").glob("**/train/%s/weights/best.pt" % tag))
            b = cands[-1] if cands else (out_dir / "weights" / "best.pt")
            print("[skip] %s 已完成，best=%s" % (tag, b))
            if b.is_file():
                prev = str(b)
            continue
        last_cands = sorted(Path("runs").glob("**/train/%s/weights/last.pt" % tag))
        resume = bool(last_cands) and last_cands[-1].is_file() \
            and not any(d.is_file() for d in done_marks)
        print("=" * 64)
        print("Stage %d/3 | imgsz=%d epochs=%d lr=%s batch=%d" % (i + 1, imsz[i], epochs[i], lrs[i], batches[i]))
        print("start: %s" % ("<断点续训 %s>" % last_cands[-1] if resume else prev))
        print("=" * 64)
        if resume:
            model = YOLO(str(last_cands[-1]))
            model.train(resume=True)      # 用 checkpoint 里保存的参数原样续跑
            t = model.trainer
            save_dir = Path(getattr(t, "save_dir", out_dir))
            best = Path(getattr(t, "best", save_dir / "weights" / "best.pt"))
            (save_dir / "PHASE_DONE").touch()
            try:
                done.touch()
            except OSError:
                pass
            print("[stage %s] save_dir=%s best=%s" % (tag, save_dir, best))
            prev = str(best)
            continue
        model = YOLO(prev)
        kw = dict(
            data=str(yaml_path), imgsz=imsz[i], epochs=epochs[i], batch=batches[i],
            optimizer=args.optimizer, lr0=lrs[i], lrf=0.01, weight_decay=0.005,
            cos_lr=True, warmup_epochs=5, patience=args.patience,
            box=7.5, dfl=1.5, close_mosaic=args.close_mosaic,
            rect=False, val=True, save=True, workers=args.workers,
            device=args.device, verbose=True,
            mosaic=0.6, mixup=0.15, copy_paste=0.1,
            degrees=5.0, translate=0.15, scale=0.5, fliplr=0.5, flipud=0.0,
            hsv_h=0.08, hsv_s=0.75, hsv_v=0.60,
            grayscale=0.15, erasing=0.25,
            label_smoothing=0.05, auto_augment=None,
            seed=args.seed_int, plots=False,
            project=args.project, name=tag, exist_ok=True,
        )
        # 自动过滤当前 ultralytics 版本不认识的参数（不同版本参数集不同，避免 SyntaxError）
        from ultralytics.cfg import get_cfg as _get_cfg
        _valid = set(vars(_get_cfg()).keys())
        _dropped = sorted(k for k in kw if k not in _valid)
        if _dropped:
            print("[warn] 你的 ultralytics 版本不支持这些参数，已自动忽略: %s" % ", ".join(_dropped))
        kw = {k: v for k, v in kw.items() if k in _valid}
        model.train(**kw)
        t = model.trainer
        save_dir = Path(getattr(t, "save_dir", out_dir))
        best = Path(getattr(t, "best", save_dir / "weights" / "best.pt"))
        (save_dir / "PHASE_DONE").touch()
        try:
            done.touch()
        except OSError:
            pass  # out_dir 在真实 save_dir 之外时不存在也无关紧要（跳过检测走 glob）
        print("[stage %s] save_dir=%s best=%s" % (tag, save_dir, best))
        prev = str(best)

    print("\n[seed] 全部完成，最终 RGB 种子 = %s" % prev)
    print("（注意：真实路径在 runs/detect/runs/train/ 下，下面命令已按此给出）")
    print("[next] 融合训练：")
    print("  python bisai_v5/v5_train_fusion.py --seed %s \\" % prev)
    print("      --data %s --imgsz 1024 --epochs 60 --batch 4 --lr 1e-3 \\"
          % (args.data or "/root/autodl-tmp/data_full"))
    print("      --freeze-backbone --warmup-epochs 10 --name v5_fusion_a")


if __name__ == "__main__":
    main()
