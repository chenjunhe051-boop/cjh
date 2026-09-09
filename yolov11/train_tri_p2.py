import warnings, os
warnings.filterwarnings('ignore')
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO('ultralytics/cfg/models/multimodal/Multi-Tri-CIFusion3-m-p2.yaml')
    model.load('weights/yolo11m_p2_init.pt')   # 需先运行 make_tri_init 生成

    model.train(
        data='dataset/data_city_multimodal.yaml',
        cache=False,
        imgsz=832,
        scale=0.2,
        epochs=230,
        patience=0,
        batch=8,
        close_mosaic=15,
        workers=8,
        optimizer='SGD',
        lr0=0.01,
        hsv_h=0.0, hsv_s=0.0, hsv_v=0.0,
        project='runs/city',
        name='tri-cifusion3-yolo11m-p2',
    )
