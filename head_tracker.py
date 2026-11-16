"""
Head Tracker Module - MediaPipe Face Mesh ile Kafa ve Derinlik Tespiti
----------------------------------------------------------------------
Bu modül, webcam görüntüsünden MediaPipe kullanarak göz merkezlerini 
ve gözler arası uzaklıktan (IPD) türetilen Z (derinlik) bilgisini hesaplar.
MediaPipe Solutions ve MediaPipe Tasks API'lerinin her ikisini de destekler.
"""

import builtins
import os
import time
import urllib.request
import cv2
import numpy as np


def _import_mediapipe_without_audio():
    """Load MediaPipe Vision without initializing the optional audio backend."""
    original_import = builtins.__import__

    def import_without_sounddevice(name, *args, **kwargs):
        if name == "sounddevice":
            raise ImportError("Audio backend is not used by this application.")
        return original_import(name, *args, **kwargs)

    builtins.__import__ = import_without_sounddevice
    try:
        import mediapipe as mediapipe
    finally:
        builtins.__import__ = original_import
    return mediapipe


mp = _import_mediapipe_without_audio()


class HeadTracker:
    def __init__(self, camera_id=0, width=640, height=480, real_ipd_cm=6.3):
        """
        :param camera_id: Kamera aygıt kimliği (varsayılan: 0)
        :param width: Kamera genişliği (piksel)
        :param height: Kamera yüksekliği (piksel)
        :param real_ipd_cm: Ortalama insan iki göz bebek arası mesafesi (cm)
        """
        self.camera_id = camera_id
        self.width = width
        self.height = height
        self.real_ipd_cm = real_ipd_cm

        # Pinhole Kamera Odak Uzaklığı Tahmini (f_px ~= width * 0.85)
        self.focal_length_px = width * 0.85

        self.cap = None
        self.mode = None  # 'solutions' veya 'tasks'

        # MediaPipe API Mod Tespiti ve Başlatılması
        if hasattr(mp, 'solutions') and hasattr(mp.solutions, 'face_mesh'):
            self.mode = 'solutions'
            self.mp_face_mesh = mp.solutions.face_mesh
            self.face_mesh = self.mp_face_mesh.FaceMesh(
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=0.6,
                min_tracking_confidence=0.6
            )
        else:
            self.mode = 'tasks'
            self._init_tasks_api()

    def _init_tasks_api(self):
        """MediaPipe 1.0+ Tasks API için FaceLandmarker yapılandırması."""
        model_path = os.path.join(os.path.dirname(__file__), "face_landmarker.task")
        if not os.path.exists(model_path):
            print("[INFO] MediaPipe FaceLandmarker model dosyası indiriliyor...")
            url = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
            urllib.request.urlretrieve(url, model_path)
            print("[INFO] Model indirme tamamlandı.")

        from mediapipe.tasks import python
        from mediapipe.tasks.python import vision

        base_options = python.BaseOptions(model_asset_path=model_path)
        # VIDEO modu: önceki karedeki yüzü takip eder; IMAGE moduna göre daha hızlı ve kararlı
        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.VIDEO,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False,
            num_faces=1
        )
        self.landmarker = vision.FaceLandmarker.create_from_options(options)

    def start_camera(self):
        """Kamerayı başlatır ve çözünürlüğü ayarlar."""
        self.cap = cv2.VideoCapture(self.camera_id)
        if not self.cap.isOpened():
            raise RuntimeError(f"Kamera ({self.camera_id}) açılamadı! Lütfen cihaz bağlantısını kontrol edin.")
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        # Sürücü kare biriktirmesin: her zaman en yeni kare işlenir (gecikme azalır)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        # MJPG + 30 fps iste (sürücü desteklemezse sessizce yok sayılır)
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        self._t0 = time.monotonic()
        self._last_ts_ms = -1

    def release(self):
        """Kaynakları serbest bırakır."""
        if self.cap and self.cap.isOpened():
            self.cap.release()
        if self.mode == 'solutions':
            self.face_mesh.close()
        elif self.mode == 'tasks':
            self.landmarker.close()

    def process_frame(self):
        """
        Kameradan tek bir kare işler.
        :return: (success, frame, raw_pose)
        """
        if self.cap is None or not self.cap.isOpened():
            return False, None, None

        ret, frame = self.cap.read()
        # Yakalama zamanı: çıkarım süresi ölçüm yaşına sayılmasın (gecikme tahmini doğru olsun)
        t_capture = time.monotonic()
        if not ret:
            return False, None, None

        # Doğal hareket hissi için aynalama
        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        landmarks = None

        if self.mode == 'solutions':
            results = self.face_mesh.process(rgb_frame)
            if results.multi_face_landmarks:
                landmarks = results.multi_face_landmarks[0].landmark
        else:
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            # VIDEO modu kesin artan milisaniye zaman damgası ister
            ts_ms = max(int((time.monotonic() - self._t0) * 1000), self._last_ts_ms + 1)
            self._last_ts_ms = ts_ms
            results = self.landmarker.detect_for_video(mp_image, ts_ms)
            if results.face_landmarks:
                landmarks = results.face_landmarks[0]

        if landmarks is None:
            return True, frame, None

        # Göz merkezleri = göz köşelerinin ortası. Göz bebekleri (iris 468/473) kullanılmaz:
        # kullanıcı ekranda başka yere bakınca kayarlar ve görüntüyü titretirler.
        l1 = np.array([landmarks[33].x * w, landmarks[33].y * h])
        l2 = np.array([landmarks[133].x * w, landmarks[133].y * h])
        r1 = np.array([landmarks[362].x * w, landmarks[362].y * h])
        r2 = np.array([landmarks[263].x * w, landmarks[263].y * h])
        left_eye_px = (l1 + l2) / 2.0
        right_eye_px = (r1 + r2) / 2.0

        # 1. Kafa Merkezi Piksel Koordinatları
        head_center_px = (left_eye_px + right_eye_px) / 2.0

        # 2. Göz Mesafesi (IPD in pixels)
        ipd_px = np.linalg.norm(left_eye_px - right_eye_px)

        # 3. Pinhole Kamera Modeli ile Derinlik (Z in cm)
        depth_z_cm = (self.focal_length_px * self.real_ipd_cm) / max(ipd_px, 1e-5)

        raw_pose = {
            'x_px': float(head_center_px[0]),
            'y_px': float(head_center_px[1]),
            'z_cm': float(depth_z_cm),
            'ipd_px': float(ipd_px),
            't_capture': t_capture,
            'left_eye_px': (int(left_eye_px[0]), int(left_eye_px[1])),
            'right_eye_px': (int(right_eye_px[0]), int(right_eye_px[1])),
            'head_center_px': (int(head_center_px[0]), int(head_center_px[1]))
        }

        return True, frame, raw_pose
