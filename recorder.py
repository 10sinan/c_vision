"""
Video kaydı: sahne (solda) + kamera görüntüsü (sağda) yan yana, tek mp4 dosyası.

Render döngüsü yalnızca ham kareleri kuyruğa atar; birleştirme (compose), yeniden boyutlandırma ve
diske yazma ayrı bir iş parçacığında yapılır. Kuyruk dolarsa kare atlanır (render asla beklemez).
Süre doğruluğu: kareler gerçek zaman damgasına göre 30 fps ızgarasına oturtulur (yavaşlayınca
son kare tekrarlanır), böylece video gerçek hızda oynar.
"""

import os
import queue
import threading
import time
from datetime import datetime

import cv2
import numpy as np

FPS = 30
SCENE_W, PANEL_H = 960, 540          # sahne paneli 16:9
CAM_W = 720                          # kamera paneli 4:3 -> 720x540
OUT_W, OUT_H = SCENE_W + CAM_W, PANEL_H
RECORDINGS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "kayitlar")
QUEUE_SIZE = 8

_GREEN = (0, 255, 0)
_WHITE = (255, 255, 255)


def _fit(img, width, height):
    """Görüntüyü oranını koruyarak (width, height) alanına sığdırır, kalan yeri siyah bırakır."""
    h, w = img.shape[:2]
    k = min(width / w, height / h)
    nw, nh = max(1, int(round(w * k))), max(1, int(round(h * k)))
    out = np.zeros((height, width, 3), dtype=np.uint8)
    x, y = (width - nw) // 2, (height - nh) // 2
    out[y:y + nh, x:x + nw] = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
    return out


def compose_frame(scene_rgb, cam_bgr, pose=None, z_cm=None):
    """
    :param scene_rgb: (h, w, 3) RGB sahne karesi (renderer.grab_frame)
    :param cam_bgr: (h, w, 3) BGR kamera karesi | None (fare modu)
    :param pose: göz konumlarını içeren poz sözlüğü | None (yüz bulunamadı)
    :param z_cm: gösterilecek mesafe | None
    :return: (OUT_H, OUT_W, 3) BGR kare
    """
    left = _fit(cv2.cvtColor(np.ascontiguousarray(scene_rgb), cv2.COLOR_RGB2BGR), SCENE_W, PANEL_H)
    if cam_bgr is None:
        right = np.zeros((PANEL_H, CAM_W, 3), dtype=np.uint8)
        cv2.putText(right, "mouse mode (no camera)", (40, PANEL_H // 2), cv2.FONT_HERSHEY_SIMPLEX,
                    0.9, _WHITE, 2, cv2.LINE_AA)
    else:
        cam = cam_bgr.copy()
        if pose is not None:
            for key in ("left_eye_px", "right_eye_px"):
                cv2.circle(cam, pose[key], 6, _GREEN, 2)
            cv2.line(cam, pose["left_eye_px"], pose["right_eye_px"], _GREEN, 1)
        right = _fit(cam, CAM_W, PANEL_H)
        if z_cm is not None:
            cv2.putText(right, f"{z_cm:.0f} cm", (16, 36), cv2.FONT_HERSHEY_SIMPLEX, 1.0, _GREEN, 2, cv2.LINE_AA)
    return np.hstack([left, right])


class Recorder:
    """V tuşuyla aç/kapa. start()/stop() ana iş parçacığından; capture() her render karesinde çağrılır."""

    def __init__(self, directory=RECORDINGS_DIR):
        self.directory = directory
        self.active = False
        self.path = None
        self.dropped = 0
        self.written = 0
        self._queue = None
        self._thread = None
        self._t0 = 0.0
        self._last_capture = 0.0

    def start(self):
        os.makedirs(self.directory, exist_ok=True)
        self.path = os.path.join(self.directory, datetime.now().strftime("parallax_%Y%m%d_%H%M%S.mp4"))
        self._queue = queue.Queue(maxsize=QUEUE_SIZE)
        self.dropped = self.written = 0
        self._t0 = time.monotonic()
        self._last_capture = 0.0
        self.active = True
        self._thread = threading.Thread(target=self._run, args=(self.path,), daemon=True)
        self._thread.start()
        return self.path

    def due(self, now=None):
        """Yeni kare yakalamanın zamanı geldi mi? (FPS sınırı; gereksiz glReadPixels'i önler)"""
        now = time.monotonic() if now is None else now
        return self.active and now - self._last_capture >= 1.0 / FPS - 0.002

    def capture(self, scene_rgb, cam_bgr=None, pose=None, z_cm=None):
        now = time.monotonic()
        self._last_capture = now
        try:
            self._queue.put_nowait((now - self._t0, scene_rgb, cam_bgr, pose, z_cm))
        except queue.Full:
            self.dropped += 1

    def elapsed(self):
        return time.monotonic() - self._t0 if self.active else 0.0

    def stop(self):
        """Kaydı bitirir (kuyruğu boşaltıp dosyayı kapatır). :return: dosya yolu"""
        if not self.active:
            return None
        self.active = False
        self._queue.put(None)         # bitiş işareti (kuyruk doluysa yer açılana kadar bekler)
        self._thread.join(timeout=30)
        return self.path

    def _run(self, path):
        writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), FPS, (OUT_W, OUT_H))
        if not writer.isOpened():
            print("[HATA] Video dosyası açılamadı:", path)
            while self._queue.get() is not None:   # kuyruğu boşalt ki ana iş parçacığı takılmasın
                pass
            return
        written = 0
        try:
            while True:
                item = self._queue.get()
                if item is None:
                    break
                t, scene_rgb, cam_bgr, pose, z_cm = item
                frame = compose_frame(scene_rgb, cam_bgr, pose, z_cm)
                # Gerçek zamana oturt: bu zamana kadar gereken kare sayısına tamamla
                target = int(t * FPS) + 1
                while written < target:
                    writer.write(frame)
                    written += 1
        finally:
            writer.release()
            self.written = written
