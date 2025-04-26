import cv2
from ultralytics import FastSAM

# FastSAM modelini yükle
model = FastSAM('yolov10b.pt')  # Model dosyasının yolunu belirtin

# Kamerayı aç
cap = cv2.VideoCapture(0)  # 0, varsayılan kamerayı ifade eder

while True:
    # Kameradan bir kare oku
    ret, frame = cap.read()
    if not ret:
        print("Kameradan görüntü alınamadı.")
        break

    # Görüntüyü FastSAM ile işle
    results = model.predict(frame)

    # Sonuçları görselleştir
    for result in results:
        masks = result.masks  # Segmentasyon maskeleri
        for mask in masks:
            # Maskeyi uygulama (örneğin, yeşil renk ile)
            colored_mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR) * (0, 255, 0)
            frame = cv2.addWeighted(frame, 1, colored_mask, 0.5, 0)

    # Sonucu göster
    cv2.imshow('Kamera Görüntüsü', frame)

    # 'q' tuşuna basıldığında döngüden çık
    if cv2.waitKey() & 0xFF == ord('q'):
        break

# Kamerayı serbest bırak
cap.release()
cv2.destroyAllWindows()
