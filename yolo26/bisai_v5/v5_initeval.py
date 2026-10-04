# -*- coding: utf-8 -*-
"""v5_initeval -- 融合前后对照验证（自包含，兼容 3 通道和 9 通道 stem 的种子）。

两条腿使用完全相同的 9 通道数据管线：
  参考腿 = 种子 + 融合模块 + inject warmup=0
    - 若种子是纯 RGB 模型（3 通道 stem）：该腿逐比特等于原种子（朋友的原始设计）；
    - 若种子是老的三模态拼接模型（9 通道 stem，如你的 y26x_o365）：该腿经 legacy
      映射后逐值等于原模型 —— 地板就是你拿 48 分的模型本身。
  对比腿 = 训练后的融合 best.pt。

用法（在 YOLO-Master 目录下）：
  # 训练前（单腿）：确认参考腿的分数 ≈ 你种子当年的 val 分数（你的是 0.913 左右）
  python bisai_v5/v5_initeval.py --seed runs/detect/runs/y26x_o365/weights/best.pt --imgsz 1024

  # 训练后（两腿对比）：
  python bisai_v5/v5_initeval.py --seed runs/detect/runs/y26x_o365/weights/best.pt \
      --model runs/train/v5_fusion_a/weights/best.pt --imgsz 1024
"""
import argparse
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import v5_channels                     # noqa: E402  (安装即自证 + 中和老补丁)
from v5_channels import install        # noqa: E402

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
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", required=True, help="种子权重（你的 y26x_o365 best.pt）")
    ap.add_argument("--model", default=None, help="训练后的融合 best.pt（不传则只跑参考腿）")
    ap.add_argument("--data", default=os.environ.get("DATA_FULL", ""))
    ap.add_argument("--imgsz", type=int, default=1024)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--no-amp", action="store_true")
    args = ap.parse_args()

    install()
    from ultralytics import YOLO
    import ultralytics.data.dataset as ds_mod
    import ultralytics.data.build as bd_mod
    from v5_dataset import RarePasteDatasetV4
    from v5_model import apply_fusion_v5, set_inject_warmup

    data = find_data(args.data)
    if data is None:
        sys.exit("[error] 找不到数据目录（需要 images/ ir/ depth/ splits/），用 --data 指定")
    print("[initeval] data:", data)
    RarePasteDatasetV4.TRAIN_IR = data / "ir"
    RarePasteDatasetV4.TRAIN_DEPTH = data / "depth"
    RarePasteDatasetV4.VAL_IR = [d for d in (data / "ir_val", data / "ir") if d.is_dir()]
    RarePasteDatasetV4.VAL_DEPTH = [d for d in (data / "depth_val", data / "depth") if d.is_dir()]
    yaml_path = write_yaml(data, HERE / "v5_eval_data.yaml")

    ds_mod.YOLODataset = RarePasteDatasetV4      # 两条腿都用 9 通道管线
    bd_mod.YOLODataset = RarePasteDatasetV4
    print("[initeval] val loader: 9ch tri-modal")

    amp = not args.no_amp

    def run_val(model, tag):
        print("[initeval] validating (%s) @ %d ..." % (tag, args.imgsz))
        metrics = model.val(data=yaml_path, imgsz=args.imgsz, batch=args.batch,
                            split="val", device=os.environ.get("V5_DEVICE", "0"),
                            verbose=True, plots=False, amp=amp)
        m = metrics.box
        print("  -> %s: mAP50-95=%.5f  mAP50=%.5f" % (tag, m.map, m.map50))
        return float(m.map), float(m.map50)

    seed_path = Path(args.seed)
    if not seed_path.exists():
        sys.exit("[error] seed not found: %s" % seed_path)

    # 参考腿：种子 + 融合模块 + warmup=0（对 3ch/9ch stem 都逐值等于种子本身）
    ref = YOLO(str(seed_path))
    apply_fusion_v5(ref.model)
    set_inject_warmup(ref.model, 0.0)
    print("[initeval] 参考腿：种子 + 融合模块 + warmup=0（应等于种子本身）")
    ref_map, ref_map50 = run_val(ref, "reference (fusion@0)")

    if args.model is None:
        print("=" * 60)
        print("参考腿 mAP50-95=%.5f mAP50=%.5f" % (ref_map, ref_map50))
        print("对比种子当年的 val 分数（你的 y26x_o365 约 0.913）：接近即正常。")
        print("tensor 级 identity 请跑: python bisai_v5/v5_channels.py --forwardtest --model %s"
              % seed_path)
        print("=" * 60)
        return

    model_path = Path(args.model)
    if not model_path.exists():
        sys.exit("[error] model not found: %s" % model_path)
    m = YOLO(str(model_path))
    apply_fusion_v5(m.model)            # 幂等：装派发器 + legacy 检测
    set_inject_warmup(m.model, 1.0)     # 已训练模型评估 = 注入全开
    print("[initeval] 训练模型评估采用 warmup=1.0（注入全开）")
    fuse_map, fuse_map50 = run_val(m, "fusion trained (%s)" % model_path.parent.parent.name)

    print("=" * 60)
    print("reference : mAP50-95=%.5f  mAP50=%.5f" % (ref_map, ref_map50))
    print("FUSION    : mAP50-95=%.5f  mAP50=%.5f" % (fuse_map, ref_map50))
    print("delta     : %+.5f  (mAP50 %+.5f)" % (fuse_map - ref_map, fuse_map50 - ref_map50))
    print("=" * 60)
    if fuse_map > ref_map + 1e-4:
        print("-> 融合确实涨了 %+.5f，可以进 Phase B / 提交。" % (fuse_map - ref_map))
    else:
        print("-> 融合没涨过参考：调整 warmup / lr / modal-drop 后再试，或直接用旧提交保底。")


if __name__ == "__main__":
    main()
