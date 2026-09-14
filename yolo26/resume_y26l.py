from ultralytics import YOLO
YOLO('runs/detect/runs/city26/tri-yolo26l-p2-os190/weights/last.pt').train(resume=True)
