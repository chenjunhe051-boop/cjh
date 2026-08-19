"""
train_max.py —— DEYOLO-P2 训练脚本（RTX 5090 32G 专用）
P2检测头：stride=4 特征图，专门抓 ball/sign/uav/light 等小目标

改进清单（保留）：
  ✅ 深度三通道差异化编码（对数+线性+梯度）
  ✅ VariFocal Loss 换回 BCE（exp2 验证，更稳定）
  ✅ 标准 CIoU（不用 WIoU，防过拟合）
  ✅ Modality Dropout + IR噪声 + 深度空洞
  ✅ EMA（ultralytics 内置，自动保存）
  ✅ HSV 仅用于 RGB
  ✅ 保守增强策略（exp2_640aug 验证过，40+ 分）

v3 修复：
  🔧 WIoU 换回 CIoU（v4 过拟合根因）
  🔧 BCE + 线性LR + 标准loss权重 + 简化增强
  🔧 冷启动 yolov8x.pt
"""
import torch
from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO("ultralytics/models/v8/DEYOLO-p2.yaml")

    # ========== 冷启动：加载 yolov8x.pt ==========
    print("冷启动：加载 yolov8x.pt ...")
    ckpt = torch.load("yolov8x.pt", map_location='cuda', weights_only=False)
    pretrained_dict = ckpt['model'].state_dict()
    model_dict = model.model.state_dict()

    matched_dict = {}
    for k, v in pretrained_dict.items():
        if k in model_dict:
            if v.shape == model_dict[k].shape:
                matched_dict[k] = v
            else:
                print(f"Shape mismatch: {k} | pretrained {v.shape} vs model {model_dict[k].shape}")
                # backbone2 需要 6 通道输入，从 3 通道预训练权重扩展
                if 'backbone2' in k and 'conv.weight' in k and v.shape[1] == 3 and model_dict[k].shape[1] == 6:
                    new_v = torch.zeros_like(model_dict[k])
                    new_v[:, :3, :, :] = v[:, :3, :, :]
                    new_v[:, 3:, :, :] = v[:, :3, :, :]
                    matched_dict[k] = new_v
                    print(f"  -> Expanded 3ch -> 6ch")

    model.model.load_state_dict(matched_dict, strict=False)
    print(f"冷启动加载了 {len(matched_dict)}/{len(model_dict)} 层预训练权重")

    # ========== 训练 ==========
    model.train(
        data="mydata.yaml",
        epochs=200,               # exp2 180轮到最佳，P2收敛稍慢给200
        imgsz=1280,
        batch=8,                  # RTX 5090 32G，P2多15%显存
        device="cuda",
        cache=True,               # 90GB RAM 够用，提速 ~30%
        augment=True,
        workers=8,
        amp=True,
        patience=30,              # 30轮不涨自动停
        project="my_project",
        name="exp_v8x_1280_p2",

        # 每 10 epoch 存权重，中断不丢进度
        save_period=10,
        # 画验证集预测图，监控训练
        plots=True,
        # 方形训练
        rect=False,

        # ========== 损失权重（回归 exp2 标准值）==========
        box=7.5,                  # 标准 YOLO box 权重
        cls=0.5,                  # 标准 BCE cls 权重（exp2 验证，44分）
        dfl=1.5,                  # 标准 DFL 权重（不是 3.0）

        # ========== 正则化 ==========
        weight_decay=0.0005,
        label_smoothing=0.0,      # 不用 label smoothing（exp2 也没用）

        # ========== 数据增强（exp2 保守策略）==========
        mixup=0.0,                # 不用 mixup
        mosaic=1.0,
        close_mosaic=20,          # 最后 20 轮精调 (180轮mosaic + 20轮干净图)

        # 几何增强（保守：不旋转不剪切，exp2 验证过）
        scale=0.5,                # 50%~150% 缩放（不是 0.9）
        degrees=0.0,              # 不旋转
        translate=0.1,
        shear=0.0,                # 不剪切
        flipud=0.0,               # 不垂直翻转（不是 0.5）
        fliplr=0.5,               # 水平翻转

        # HSV 颜色增强（仅对 RGB，不对 IR/Depth）
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,

        # ========== 学习率（exp2 验证过的线性衰减）==========
        cos_lr=False,             # 线性衰减（不用 cos）
        lr0=0.01,                 # 初始学习率
        lrf=0.01,                 # 最终 lr = lr0 * lrf = 1e-4
        warmup_epochs=3,
        warmup_momentum=0.8,
        warmup_bias_lr=0.1,
    )

    print("\n训练完成！")
    print("最佳权重: my_project/exp_v8x_1280_v3/weights/best.pt")
    print("推理时 prediction_tta.py 会自动使用 EMA 权重")
