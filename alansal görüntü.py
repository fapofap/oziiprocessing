from ultralytics import YOLO
import cv2
import cvzone
import time
import math
import numpy as np
import threading
import winsound
import platform


# Alarm sesi çalma fonksiyonu (ayrı bir thread'de çalışacak)
def play_alarm():
    print("UYARI: İnsan algılandı!")

    # İşletim sistemine göre ses çalma
    if platform.system() == "Windows":
        for _ in range(3):  # 3 kez bip sesi çal
            winsound.Beep(1000, 500)  # 1000 Hz, 500 ms
            time.sleep(0.2)
    else:  # Linux için alternatif
        try:
            import os
            for _ in range(3):
                os.system('play -nq -t alsa synth 0.5 sine 1000')  # Linux'ta 'sox' paketi gerekli
                time.sleep(0.2)
        except:
            print("Ses çalma için 'sox' paketi gerekli veya bu platformda desteklenmiyor.")


# Ana uygulama
class HumanDetector:
    def __init__(self, model_path="yolo11s-seg.pt", video_source=0):
        self.cap = cv2.VideoCapture(video_source)
        self.cap.set(3, 1280)  # Tam genişlik
        self.cap.set(4, 720)  # Tam yükseklik

        self.model = YOLO(model_path)
        self.classNames = ['Person']  # Basit bir sınıf adı

        # FPS hesaplama için değişkenler
        self.prev_time = time.time()
        self.fps = 0

        # ROI alanı (algılama yapılacak bölge)
        self.roi_x, self.roi_y = 50, 50
        self.roi_width, self.roi_height = 500, 500

        # Alarm durumu
        self.alarm_active = False
        self.alarm_thread = None
        self.alarm_cooldown = 5  # Alarmlar arası bekleme süresi (saniye)
        self.last_alarm_time = 0

    def trigger_alarm(self):
        """Alarm sesini tetikler - eğer cooldown süresi geçtiyse"""
        current_time = time.time()
        if not self.alarm_active and (current_time - self.last_alarm_time) > self.alarm_cooldown:
            self.alarm_active = True
            self.last_alarm_time = current_time

            # Yeni bir thread'de alarm sesini çal
            self.alarm_thread = threading.Thread(target=self.run_alarm)
            self.alarm_thread.daemon = True  # Ana program kapanınca thread de kapanır
            self.alarm_thread.start()

    def run_alarm(self):
        """Alarm sesini çalıştırır ve alarm durumunu yönetir"""
        play_alarm()
        self.alarm_active = False

    def run(self):
        """Ana detection döngüsü"""
        while True:
            success, img = self.cap.read()
            if not success:
                break

            # FPS hesaplama
            current_time = time.time()
            self.fps = 1 / (current_time - self.prev_time)
            self.prev_time = current_time

            # Belirtilen ROI'yi (Region of Interest) al
            roi = img[self.roi_y:self.roi_y + self.roi_height, self.roi_x:self.roi_x + self.roi_width].copy()

            # Sadece ROI üzerinde model tahmini yap
            results = self.model.predict(source=roi, conf=0.5, verbose=False)

            # ROI etrafına dikdörtgen çiz
            cv2.rectangle(img, (self.roi_x, self.roi_y),
                          (self.roi_x + self.roi_width, self.roi_y + self.roi_height),
                          (0, 255, 0), 2)

            # İnsan algılandı mı? (Alarm için)
            person_detected = False

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
                    class_name = self.classNames[0] if cls >= len(self.classNames) else self.classNames[cls]

                    # İnsan algılandı
                    if class_name == 'Person' and conf > 0.5:
                        person_detected = True

                    # Eğer metin ROI içerisine sığacaksa ekle
                    if y1 > 35:  # Metin için yeterli alan var mı?
                        cvzone.putTextRect(roi, f'{class_name} {conf}', (max(0, x1), max(35, y1)), scale=1, thickness=1)

            # Eğer insan algılandıysa ve alarm aktif değilse, alarm tetikle
            if person_detected:
                self.trigger_alarm()
                # Alarm durumunu belirten bir uyarı metni göster
                cv2.putText(img, "UYARI: Insan Algilandi!", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            # ROI'ye FPS'i göster (metin için alan varsa)
            if self.roi_height > 40:
                cvzone.putTextRect(roi, f'FPS: {int(self.fps)}', (5, 20), scale=0.6, thickness=1)

            # İşlenmiş ROI'yi orijinal görüntüye geri koy
            img[self.roi_y:self.roi_y + self.roi_height, self.roi_x:self.roi_x + self.roi_width] = roi

            # Alarm durumunu ekranda göster
            if self.alarm_active:
                cv2.putText(img, "ALARM AKTIF", (self.roi_x + self.roi_width + 10, self.roi_y + 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            cv2.imshow("Insan Algilama ve Alarm Sistemi", img)
            if cv2.waitKey(1) & 0xFF == ord('q'):  # q tuşu ile çıkış
                break

        self.cap.release()
        cv2.destroyAllWindows()


# Ana programı başlat
if __name__ == "__main__":
    detector = HumanDetector()
    detector.run()
