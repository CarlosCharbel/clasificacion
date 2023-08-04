from ultralytics import YOLO


model = YOLO('yolov8n-cls.pt')

model.train(data='/home/frisaros/clasificacion/train_dataset',epochs=20,imgsz=64)