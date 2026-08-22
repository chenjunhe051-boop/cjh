"""
resume_train.py —— 从 exp_improved_640_afitd2 断点续训
"""
import os

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

import torch
from ultralytics import YOLO

if __name__ == '__main__':
    # 加载 last.pt 断点权重
    model = YOLO("/root/autodl-tmp/my_project/exp_improved_640_afitd2/weights/last.pt")

    # resume=True 会自动恢复优化器状态、学习率调度、当前 epoch 等
    model.train(
        resume=True,  # ← 关键：断点续训
        data="mydata.yaml",
        epochs=30,  # 总目标轮数（如果已经跑了 N 轮，会继续跑到 30）
        imgsz=640,
        batch=16,
        device="cuda",
        cache=True,
        augment=True,
        workers=8,
        amp=True,
        patience=10,
        project="my_project",
        name="exp_improved_640_afitd2",  # 必须和原实验名一致，才能写到同一目录
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

    print("\n续训完成！")
    print("最佳权重: /root/autodl-tmp/my_project/exp_improved_640_afitd2/weights/best.pt")