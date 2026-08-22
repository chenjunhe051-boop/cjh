"""
train_improved_640.py —— DEYOLO-P2 + AFITDYOLO改进版 (MFFM/CAFM/MFEConv)
640分辨率, 30轮, batch=4 (48G显存+bmm适配), 与 baseline 严格对照
"""
import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

import torch
torch.use_deterministic_algorithms(False)

from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO("ultralytics/models/v8/DEYOLO-p2-AFITD.yaml")

    print("冷启动：加载 yolov8s.pt ...")
    ckpt = torch.load("yolov8s.pt", map_location='cuda', weights_only=False)
    pretrained_dict = ckpt['model'].state_dict()
    model_dict = model.model.state_dict()

    matched_dict = {}
    for k, v in pretrained_dict.items():
        if k in model_dict:
            if v.shape == model_dict[k].shape:
                matched_dict[k] = v
            else:
                print(f"Shape mismatch: {k} | pretrained {v.shape} vs model {model_dict[k].shape}")
                if 'backbone2' in k and 'conv.weight' in k and v.shape[1] == 3 and model_dict[k].shape[1] == 6:
                    new_v = torch.zeros_like(model_dict[k])
                    new_v[:, :3, :, :] = v[:, :3, :, :]
                    new_v[:, 3:, :, :] = v[:, :3, :, :]
                    matched_dict[k] = new_v
                    print(f" -> Expanded 3ch -> 6ch")

    model.model.load_state_dict(matched_dict, strict=False)
    print(f"冷启动加载了 {len(matched_dict)}/{len(model_dict)} 层预训练权重")

    model.train(
        data="mydata.yaml",
        epochs=30,
        imgsz=640,
        batch=16,             # 48G + 原版bmm 适配
        device="cuda",
        cache=True,
        augment=True,
        workers=8,
        amp=True,
        patience=10,
        project="my_project",
        name="exp_improved_640_afitd",
        save_period=10,
        plots=True,
        rect=False,
        box=7.5,
        cls=0.5,
        dfl=1.5,
        weight_decay=0.0005,
        label_smoothing=0.0,
        mixup=0.0,
        mosaic=1.0,
        close_mosaic=5,
        scale=0.5,
        degrees=0.0,
        translate=0.1,
        shear=0.0,
        flipud=0.0,
        fliplr=0.5,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        cos_lr=False,
        lr0=0.01,
        lrf=0.01,
        warmup_epochs=3,
        warmup_momentum=0.8,
        warmup_bias_lr=0.1,
    )

    print("\n训练完成！")
    print("最佳权重: my_project/exp_improved_640_afitd/weights/best.pt")