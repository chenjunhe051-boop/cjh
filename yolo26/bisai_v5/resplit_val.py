from pathlib import Path
import cv2, random, re
import numpy as np

df = Path("/root/autodl-tmp/data_full/splits")
train = [l.strip() for l in (df/"train.txt").read_text().splitlines() if l.strip()]
val   = [l.strip() for l in (df/"val.txt").read_text().splitlines() if l.strip()]

def stem(p):
    return re.sub(r'_os\d+$', '', Path(p).stem)

groups = {}
for p in train:
    groups.setdefault(stem(p), []).append(p)

random.seed(0)
items = list(groups.items())
random.shuffle(items)

means = {}
for s, files in items:
    img = cv2.imread(files[0])
    if img is not None:
        means[s] = float(img.mean())

mu = np.array(sorted(means.values()))
print("唯一训练图: %d 张" % len(mu))
print("亮度分布: min=%.0f p10=%.0f p25=%.0f 中位=%.0f" % (mu[0], mu[len(mu)//10], mu[len(mu)//4], mu[len(mu)//2]))
for t in (60, 80, 100, 120):
    print("  亮度<%d: %d 张" % (t, int((mu < t).sum())))

# 自适应：把"最暗的 8% 唯一图"整组挪进 val（且不少于亮度<90 的全部）
n_dark = max(len(mu) // 12, int((mu < 90).sum()))
n_dark = min(n_dark, 150)
dark_stems = set(sorted(means, key=lambda s: means[s])[:n_dark])
move_files = set()
for s in dark_stems:
    move_files.update(groups[s])

new_val = val + sorted(move_files)
new_train = [p for p in train if p not in move_files]
(df/"val.txt").write_text("\n".join(new_val) + "\n")
(df/"train.txt").write_text("\n".join(new_train) + "\n")
print("\n挪入 val: %d 张唯一图中的 %d 个文件（最暗 8%%，其所有副本已离开训练集）"
      % (len(dark_stems), len(move_files)))
print("新 val: %d 张  新 train: %d 张" % (len(new_val), len(new_train)))
