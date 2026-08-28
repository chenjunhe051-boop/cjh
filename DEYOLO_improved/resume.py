
import torch
from ultralytics import YOLO

if __name__ == '__main__':
    # 加载 last.pt 断点权重
    model = YOLO("/root/autodl-tmp/my_project/exp_improved_640_v1.6_final/weights/last.pt")

    # resume=True 会自动恢复优化器状态、学习率调度、当前 epoch 等
    model.train(
        resume=True,
        data="mydata.yaml",
        epochs=180,
        imgsz=640,
        batch=16,
        device="cuda",
        cache=True,
        augment=True,
        workers=8,
        amp=True,
        patience=30,
        project="my_project",
        name="exp_improved_640_v1.6_final",  # 改名
        save_period=20,
        plots=True,
        rect=False,
        box=7.5,
        cls=0.7,
        dfl=1.5,
        weight_decay=0.0005,
        label_smoothing=0.0,
        mixup=0.0,
        mosaic=1.0,
        close_mosaic=10,
        scale=0.5,
        degrees=0.0,
        translate=0.1,
        shear=0.0,
        flipud=0.0,
        fliplr=0.5,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        cos_lr=True,  # 改这里
        lr0=0.008,
        lrf=0.01,
        warmup_epochs=10,
        warmup_momentum=0.8,
        warmup_bias_lr=0.1,
        deterministic=False,  # ← 加这一行！
    )

    print("\n续训完成！")
    print("最佳权重: /root/autodl-tmp/my_project/exp_improved_640_v1.6_final/weights/best.pt")