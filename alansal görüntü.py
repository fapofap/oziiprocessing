from ultralytics import YOLO
import cv2
import cvzone
import time
import math
import numpy as np

cap = cv2.VideoCapture(0)  # For Webcam
cap.set(3, 1280)  # Tam genişlik
cap.set(4, 720)  # Tam yükseklik

model = YOLO("yolo11s-seg.pt")

classNames = ['Person']  # Basit bir sınıf adı

# FPS hesaplama için değişkenler
prev_time = time.time()
fps = 0

# İşlenecek bölgenin koordinatları (sol üst köşe x, y ve genişlik, yükseklik)
roi_x, roi_y = 50, 50  # Sol üst köşe başlangıç koordinatları
roi_width, roi_height = 400, 400  # ROI boyutu

while True:
    success, img = cap.read()
    if not success:
        break

    # FPS hesaplama
    current_time = time.time()
    fps = 1 / (current_time - prev_time)
    prev_time = current_time

    # Belirtilen ROI'yi (Region of Interest) al
    roi = img[roi_y:roi_y + roi_height, roi_x:roi_x + roi_width].copy()

    # Sadece ROI üzerinde model tahmini yap
    results = model.predict(source=roi, conf=0.5, verbose=False)

    # ROI etrafına dikdörtgen çiz (opsiyonel)
    cv2.rectangle(img, (roi_x, roi_y), (roi_x + roi_width, roi_y + roi_height), (0, 255, 0), 2)

    # Sonuçları sadece ROI'ye çiz
    for r in results:
        boxes = r.boxes
        for box in boxes:
            # Bounding Box
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            w_box, h_box = x2 - x1, y2 - y1
            cvzone.cornerRect(roi, (x1, y1, w_box, h_box))
            # Confidence
            conf = math.ceil((box.conf[0] * 100)) / 100
            # Class Name
            cls = int(box.cls[0])
            class_name = classNames[0] if cls >= len(classNames) else classNames[cls]

            # Eğer metin ROI içerisine sığacaksa ekle
            if y1 > 35:  # Metin için yeterli alan var mı?
                cvzone.putTextRect(roi, f'{class_name} {conf}', (max(0, x1), max(35, y1)), scale=1, thickness=1)

    # ROI'ye FPS'i göster (metin için alan varsa)
    if roi_height > 40:
        cvzone.putTextRect(roi, f'FPS: {int(fps)}', (5, 20), scale=0.6, thickness=1)

    # İşlenmiş ROI'yi orijinal görüntüye geri koy
    img[roi_y:roi_y + roi_height, roi_x:roi_x + roi_width] = roi

    cv2.imshow("İşleme Bölgesi", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):  # q tuşu ile çıkış
        break

cap.release()
cv2.destroyAllWindows()