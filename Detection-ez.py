from ultralytics import YOLO

model = YOLO("yolo11m-pose.pt")

model.predict(source="0", show=True)