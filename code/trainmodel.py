from ultralytics import YOLO

model = YOLO("runs/segment/train-3/weights/best.pt")

model.train(
    data="dataset/data.yaml",
    epochs=100,
    imgsz=640,
    batch=8
)