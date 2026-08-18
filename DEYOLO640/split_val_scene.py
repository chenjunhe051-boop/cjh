# split_val_hash.py
import os
import shutil
from pathlib import Path
import hashlib

src_root = Path("D:/learn_pytorch/DEYOLO-main/mydata")

# 清理旧的验证集（如果有的话）
for sub in ['vis_val', 'Ir_val', 'depth_val']:
    val_dir = src_root / 'images' / sub
    if val_dir.exists():
        print(f"清理旧验证集: {val_dir}")
        for f in val_dir.glob('*'):
            f.unlink()

val_label_dir = src_root / 'labels' / 'vis_val'
if val_label_dir.exists():
    for f in val_label_dir.glob('*'):
        f.unlink()

# 确保目录存在
for sub in ['vis_val', 'Ir_val', 'depth_val']:
    (src_root / 'images' / sub).mkdir(parents=True, exist_ok=True)
(src_root / 'labels' / 'vis_val').mkdir(parents=True, exist_ok=True)

# 读取训练集文件
train_rgb_dir = src_root / 'images' / 'vis_train'
rgb_files = list(train_rgb_dir.glob('*'))

print(f"训练目录: {train_rgb_dir}")
print(f"共找到 {len(rgb_files)} 个文件")

# 按 hash 取模划分 20% 做验证集
val_files = []
for f in rgb_files:
    hash_val = int(hashlib.md5(f.stem.encode()).hexdigest(), 16)
    if hash_val % 10 < 2:  # 20% (0, 1)
        val_files.append(f)

print(f"验证集: {len(val_files)} 张, 占比 {len(val_files)/len(rgb_files)*100:.1f}%")

# 复制文件到验证集
for f in val_files:
    name = f.name
    # RGB
    shutil.copy2(f, src_root / 'images' / 'vis_val' / name)
    # 红外 (Ir_train -> Ir_val)
    ir_src = src_root / 'images' / 'Ir_train' / name
    if ir_src.exists():
        shutil.copy2(ir_src, src_root / 'images' / 'Ir_val' / name)
    # 深度
    depth_src = src_root / 'images' / 'depth_train' / name
    if depth_src.exists():
        shutil.copy2(depth_src, src_root / 'images' / 'depth_val' / name)
    # 标签
    label_src = src_root / 'labels' / 'vis_train' / (f.stem + '.txt')
    if label_src.exists():
        shutil.copy2(label_src, src_root / 'labels' / 'vis_val' / (f.stem + '.txt'))

print("划分完成！请检查 vis_val 目录确认文件数量。")