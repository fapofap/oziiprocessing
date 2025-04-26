from ultralytics import YOLO

model = YOLO("yolo11m-seg.pt")

model.predict(source="0", show=True)