"""
resume.py —— 训练中断恢复脚本
用法：如果 train_max.py 跑到一半断了（OOM、断网、关机），运行这个继续
"""

from ultralytics import YOLO

if __name__ == '__main__':
    # ========== 修改这里 ==========
    # 指向你中断的那个实验的 last.pt
    # 如果目录名不一样，改成实际的
    weights_path = "my_project/exp_v8x_1280_final/weights/last.pt"
    # ==============================

    print(f"正在从 checkpoint 恢复: {weights_path}")

    # 加载 last.pt，resume=True 会自动读取之前的训练参数（epochs、lr、数据等）
    model = YOLO(weights_path)

    # resume=True：继续训练，从断掉的 epoch 开始，保留优化器状态
    model.train(resume=True)

    # 注意：resume 时会自动沿用之前的所有参数，
    # 不需要再写 data、imgsz、batch 等