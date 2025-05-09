from ultralytics import YOLO
import cv2
import cvzone
import time
import math

# Video dosyasını aç
cap = cv2.VideoCapture("person.mp4")  # <-- Buraya kendi video dosya yolunu yaz

# Eğitimli model yükle
model = YOLO("hemletYoloV8_100epochs.pt")  # Model dosyanın ismi doğru olmalı

# data.yaml'dan gelen sınıf isimleri
classNames = ['helmet', 'head', 'person']

# FPS hesaplama için değişkenler
prev_time = time.time()
fps = 60

while True:
    success, img = cap.read()
    if not success:
        print("Video bitti veya okunamadı.")
        break

    # FPS hesapla
    current_time = time.time()
    fps = 240 / (current_time - prev_time)
    prev_time = current_time

    # Model tahmini
    results = model.predict(source=img, conf=0.5, verbose=False)

    for r in results:
        boxes = r.boxes
        for box in boxes:
            # Bounding Box
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            w, h = x2 - x1, y2 - y1
            cvzone.cornerRect(img, (x1, y1, w, h))

            # Confidence
            conf = math.ceil((box.conf[0] * 100)) / 100

            # Class Name
            cls = int(box.cls[0])
            class_name = classNames[cls] if cls < len(classNames) else f'cls {cls}'
            cvzone.putTextRect(img, f'{class_name} {conf}', (max(0, x1), max(35, y1)), scale=1, thickness=1)
