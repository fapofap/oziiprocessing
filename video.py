from ultralytics import YOLO
import cv2
import cvzone
import time
import math

video_path = "person.mp4"  # Bu kısmı kendi video dosya yolunuzla değiştirin
cap = cv2.VideoCapture(video_path)

# Videodan çözünürlük bilgileri alalım (opsiyonel)
video_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
video_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print(f"Video boyutu: {video_width}x{video_height}")

model = YOLO("hemletYoloV8_100epochs.pt")

classNames = ["head without helmet","head with helmet"]  # Basit bir sınıf adı

# FPS hesaplama için değişkenler
prev_time = time.time()
fps = 60

while True:
    success, img = cap.read()
    if not success:
        print("Video bitti veya okunamadı.")
        break

    # FPS hesaplama
    current_time = time.time()
    fps = 240 / (current_time - prev_time)
    prev_time = current_time

    # Daha hızlı tahmin için parametreler ekledim
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
            class_name = classNames[0] if cls >= len(classNames) else classNames[cls]
            cvzone.putTextRect(img, f'{class_name} {conf}', (max(0, x1), max(35, y1)), scale=1, thickness=1)

    # FPS'i görüntüde göster
    cvzone.putTextRect(img, f'FPS: {int(fps)}', (20, 40), scale=1, thickness=1)

    cv2.imshow("Video", img)

    # Daha yavaş oynatmak isterseniz bu değeri artırabilirsiniz (ms cinsinden)
    wait_key = 1  # 1ms bekle (normal hız)

    if cv2.waitKey(wait_key) & 0xFF == ord('q'):  # q tuşu ile çıkış
        break

cap.release()
cv2.destroyAllWindows()