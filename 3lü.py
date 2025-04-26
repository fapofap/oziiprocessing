from ultralytics import YOLO
import cv2
import cvzone
import time
import math

# Webcam ayarları
cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

# Eğitimli model yükle
model = YOLO("yolov8n.pt")  # Buraya kendi model dosyanı koyduğundan emin ol

# data.yaml'dan gelen sınıf isimleri
classNames = ['helmet', 'head', 'person']

# FPS hesaplama için değişkenler
prev_time = time.time()
fps = 60

while True:
    success, img = cap.read()
    if not success:
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

    # FPS'i göster
    cvzone.putTextRect(img, f'FPS: {int(fps)}', (20, 40), scale=1, thickness=1)

    cv2.imshow("Image", img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
