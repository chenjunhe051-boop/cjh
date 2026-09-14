import warnings, os
warnings.filterwarnings('ignore')
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO('ultralytics/cfg/models/Multi-Tri-CIFusion3-26l-p2.yaml')
    model.load('weights/yolo26l_tri_init.pt')

    model.train(
        data='dataset/data_city_multimodal.yaml',
        cache=False,
        imgsz=832,
        scale=0.2,
        epochs=230,            # 军规v2: 空图数3→4→5随轮数递增+57.5@193>57.1@225, 190轮封顶
        patience=0,
        batch=8,               # l档约90M参数, 会自动降到4~6, 无需干预
        save_period=10,        # 军规: 每25轮存快照, 训完批量提交选优
        close_mosaic=15,
        workers=8,
        mixup=0.1,
        optimizer='MuSGD',
        lr0=0.01,
        hsv_h=0.0, hsv_s=0.0, hsv_v=0.0,
        project='runs/city26',
        name='tri-yolo26l-p2-os190',
    )
