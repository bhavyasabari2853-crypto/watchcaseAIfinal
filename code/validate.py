from ultralytics import YOLO

# Quick 1-epoch training run to validate dataset setup
model = YOLO("yolo11n-seg.pt")

model.train(
    data="dataset/data.yaml",
    epochs=1
)