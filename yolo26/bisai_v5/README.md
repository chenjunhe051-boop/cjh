# bisai_v5 —— 9 通道三模态融合（基于你朋友的 v5 文件改好，自包含）

在你朋友文件的基础上做了这些改动：
1. 去掉对 `bisai_v4` / `v5_common` 的全部依赖（你机器上没有那个目录），需要的小工具已内联；
2. 修掉 `v5_submit.py` 里 `scales = [728]` 硬编码（现在 `--scales` 真的生效）；
3. `is_7ch` 等 v4 遗留命名统一改成 `is_fused`；
4. 新增了 `v5_train_fusion.py`（朋友那边的融合训练脚本在你没拿到的 bisai_v4 里，这里补了一个自包含版，功能等价：RGB 骨干冻结 + warmup 升温 + 模态丢弃 + 稀有类粘贴）。

## 一、放哪里

把这 8 个文件传到服务器：

    /root/autodl-tmp/YOLO-Master/bisai_v5/
        v5_channels.py
        v5_dataset.py
        v5_model.py
        v5_train_fusion.py     # 新增
        v5_initeval.py
        v5_submit.py
        v5_smoke.py
        README.md（本文件）

注意：是在 `YOLO-Master` 下新建 `bisai_v5` 文件夹，所有命令也在 `YOLO-Master` 下运行。

## 二、数据目录约定

脚本期望数据长这样（不满足就用软链接凑）：

    /root/autodl-tmp/data_full/
        images/          # RGB 图
        ir/              # 红外图
        depth/           # 深度图
        splits/
            train.txt
            val.txt

如果你的数据在 /root/autodl-tmp/dataset 或 mydata 且结构一致，直接 --data 指定即可；
不行就 ln -s 把四个子目录指过去。所有脚本都支持 --data 参数。

## 二点五、冲 55 的完整路线（推荐）

0) 先确认 O365 预训练快照名（跑一次即可，记住输出里的 model 值）：
   grep -m1 "^model:" runs/detect/runs/y26x_o365/args.yaml

1) 纯 RGB 种子训练（约 10 小时，tmux 挂后台）：
   python bisai_v5/v5_train_rgbseed.py --model <上一步的名字> \
       --data /root/autodl-tmp/data_full --name rgbseed

2) tensor 级 identity（几秒钟，应 PASS）：
   python bisai_v5/v5_channels.py --forwardtest \
       --model runs/train/rgbseed_s3/weights/best.pt --imgsz 320

3) 参考腿验证（应接近该种子自己的 val 分）：
   python bisai_v5/v5_initeval.py --seed runs/train/rgbseed_s3/weights/best.pt \
       --data /root/autodl-tmp/data_full --imgsz 1024

4) 融合训练 Phase A（约 10 小时）：
   python bisai_v5/v5_train_fusion.py \
       --seed runs/train/rgbseed_s3/weights/best.pt \
       --data /root/autodl-tmp/data_full \
       --imgsz 1024 --epochs 60 --batch 4 --lr 1e-3 \
       --freeze-backbone --warmup-epochs 10 --name v5_fusion_a

5) 对照验证（delta 为正则成功）：
   python bisai_v5/v5_initeval.py --seed runs/train/rgbseed_s3/weights/best.pt \
       --model runs/train/v5_fusion_a/weights/best.pt --imgsz 1024

6) 提交：先用 --scales 728 和 --scales 1024 各跑一次本地评分(check_submit.py)，
   选高的；CONF_TH/NMS_IOU 务必用本地评分重新标定。

## 三、运行顺序（叠加老模型的备选路线，每一步都有验证，别跳步）

    cd /root/autodl-tmp/YOLO-Master

    # 0) CPU 冒烟测试：数据布局 + 融合模块 + identity 地板，几十秒
    python bisai_v5/v5_smoke.py

    # 1) 数据管线自检：9 通道管线的 ch0..2 必须与 3 通道逐比特一致
    python bisai_v5/v5_channels.py --selftest --data /root/autodl-tmp/data_full --imgsz 1024

    # 2) 训练前 identity 验证（融合模型 warmup=0 应等于你的 y26x_o365 种子）
    python bisai_v5/v5_initeval.py \
        --seed runs/detect/runs/train/y26x_o365/weights/best.pt \
        --imgsz 1024
    # 期望最后一行：identity OK，delta < 0.01

    # 3) Phase A：冻结 RGB 骨干训融合（不会跌种子）
    python bisai_v5/v5_train_fusion.py \
        --seed runs/detect/runs/train/y26x_o365/weights/best.pt \
        --data /root/autodl-tmp/data_full \
        --imgsz 1024 --epochs 60 --batch 4 --lr 1e-3 \
        --freeze-backbone --warmup-epochs 10 --name v5_fusion_a

    # 4) 训练后对照（应高于种子）
    python bisai_v5/v5_initeval.py \
        --seed runs/detect/runs/train/y26x_o365/weights/best.pt \
        --model runs/train/v5_fusion_a/weights/best.pt --imgsz 1024

    # 5) Phase B（可选，Phase A 涨了再做）：全模型低学习率微调
    python bisai_v5/v5_train_fusion.py \
        --seed runs/detect/runs/train/y26x_o365/weights/best.pt \
        --init-from runs/train/v5_fusion_a/weights/best.pt \
        --data /root/autodl-tmp/data_full \
        --imgsz 1024 --epochs 40 --batch 4 --lr 1e-4 --name v5_fusion_b

    # 6) 生成提交压缩包（1000 张测试图，输出 zip 直接交）
    python bisai_v5/v5_submit.py runs/train/v5_fusion_a/weights/best.pt \
        --test-dir /root/autodl-tmp/test_data \
        --out /root/autodl-tmp/submit_v5

## 四、提交前必做：重新标定置信度阈值

`v5_submit.py` 里的 CONF_TH / NMS_IOU 两张表是按**你朋友的模型**调的，直接抄不一定适合你。
用你自己的 val 集跑一遍阈值搜索（你原来的调参脚本改改就能用），重点照顾小目标类
（uav、ball、sign 的 conf 通常要压得比较低）。

## 四点五、如果你的 YOLO-Master 被之前的三模态实验改过

有些代码库的 ultralytics/data/base.py 里 load_image 被写死"永远拼 9 通道"
（按 vis_->ir_/depth_ 找图）。v5_channels.install() 会自动检测并在本进程内中和它，
看到 "[v5_channels] 检测到 base.py 的 load_image 被魔改成强制9通道，已临时中和"
属正常现象 —— 这不改你的源码文件，只影响运行 v5 脚本的进程。

## 五、常见问题

- identity 不过：先跑 v5_smoke.py，它会定位是数据侧（BGR 翻转/padding）还是模型侧（零初始化/warmup）；
- 显存不够：--batch 降到 2，或 --imgsz 降到 896；
- 某阶段想重来：删掉 runs/train/<name> 整个目录再跑；
- 想禁止模态丢弃（数据侧调试）：--modal-drop 0。
