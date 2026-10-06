"""
Stage 1 Test Script - MediaPipe Kafa ve Derinlik Tespiti Visualizer
------------------------------------------------------------------
Bu betik `head_tracker.py` modülünü çalıştırır ve canlı video üzerinde:
- Sol ve sağ göz bebeklerini (Yeşil daireler)
- Kafa merkezini (Kırmızı artı işareti)
- İki göz arası piksel mesafesini (IPD)
- Kafa derinliğini (Z cinsinden cm) ekrana yazdırır.
"""

import cv2
from head_tracker import HeadTracker


def main():
    tracker = HeadTracker(camera_id=0, width=640, height=480)
    
    try:
        tracker.start_camera()
        print("[INFO] Kamera başlatıldı. Çıkmak için 'q' tuşuna basın.")
    except Exception as e:
        print(f"[HATA] Kamera başlatılamadı: {e}")
        return

    try:
        while True:
            success, frame, pose = tracker.process_frame()
            if not success or frame is None:
                print("[UYARI] Kare alınamadı veya akış bitti.")
                break

            if pose is not None:
                # Göz noktaları ve merkez çizimi
                cv2.circle(frame, pose['left_eye_px'], 5, (0, 255, 0), -1)
                cv2.circle(frame, pose['right_eye_px'], 5, (0, 255, 0), -1)
                cv2.line(frame, pose['left_eye_px'], pose['right_eye_px'], (255, 255, 0), 1)

                cx, cy = pose['head_center_px']
                cv2.drawMarker(frame, (cx, cy), (0, 0, 255), cv2.MARKER_CROSS, 15, 2)

                # Bilgi Metinleri Overlay
                cv2.putText(frame, f"Kafa X,Y: ({cx}, {cy}) px", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                cv2.putText(frame, f"Goz Mesafesi (IPD): {pose['ipd_px']:.1f} px", (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                cv2.putText(frame, f"Derinlik (Z): {pose['z_cm']:.1f} cm", (10, 90),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            else:
                cv2.putText(frame, "Yuz Tespit Edilemedi!", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            cv2.imshow("Aşama 1: Kafa ve Derinlik Tespiti", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        tracker.release()
        cv2.destroyAllWindows()
        print("[INFO] Uygulama kapatıldı.")


if __name__ == "__main__":
    main()
