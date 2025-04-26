from ultralytics import YOLO

    def main():

        model = YOLO('yolov8l.pt')
        model.train(data='data.yaml', epochs=100, patience=50, imgsz=640)
    if __name__ == "__main__":
        main()