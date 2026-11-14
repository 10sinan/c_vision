"""
Mesafe kalibrasyonu.

Derinlik (Z) tahmini pinhole modelinden gelir:  z = f_px * ipd_cm / ipd_px.
Kullanıcı kamerayı bilinen bir mesafede (ör. 60 cm) bekletirse f_px'i ters çevirip bulabiliriz:
  f_px = z_bilinen * ipd_px / ipd_cm      ->  focal_mult = f_px / frame_w

Matematik (aşağıdaki saf fonksiyonlar) OpenGL'siz test edilir; ekran akışı run_calibration()'dadır.
"""

import statistics
import time

import cv2
import numpy as np
import pygame
from pygame.locals import (QUIT, KEYDOWN, VIDEORESIZE, K_ESCAPE, K_RETURN, K_KP_ENTER,
                           K_UP, K_DOWN, K_r)
from OpenGL.GL import (glClear, glClearColor, glEnable, glBlendFunc, GL_COLOR_BUFFER_BIT,
                       GL_DEPTH_BUFFER_BIT, GL_BLEND, GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

from menu import (_blit_text, _blit_image, TITLE_COLOR, TEXT_COLOR, DIM_COLOR, SELECTED_COLOR)

MEASURE_SECONDS = 2.0
MIN_SAMPLES = 20             # ~30 Hz kamerada 2 sn ≈ 60 örnek; bunun altı güvenilmez
MAX_SPREAD = 0.05            # örneklerin medyandan sapması (MAD/medyan) bunu aşarsa "çok hareket ettin"
FOCAL_MULT_RANGE = (0.2, 3.0)
DISTANCES_CM = (40, 45, 50, 55, 60, 65, 70, 80, 90)


def focal_mult_from_measurement(ipd_px, known_z_cm, ipd_cm, frame_w):
    """:return: kamera odak çarpanı (focal_px / frame_w)"""
    if ipd_px <= 0 or ipd_cm <= 0 or frame_w <= 0:
        raise ValueError("ipd_px, ipd_cm ve frame_w pozitif olmalı")
    return known_z_cm * ipd_px / ipd_cm / frame_w


def summarize(samples):
    """
    :param samples: ölçülen ipd_px değerleri
    :return: (medyan, göreli_sapma) — sapma = MAD / medyan (aykırı değere dayanıklı)
    """
    med = statistics.median(samples)
    mad = statistics.median(abs(s - med) for s in samples)
    return med, (mad / med if med else float("inf"))


def evaluate(samples, known_z_cm, ipd_cm, frame_w):
    """
    :return: (focal_mult | None, mesaj). Yetersiz örnek, çok oynak ölçüm ya da aralık dışı sonuçta None.
    """
    if len(samples) < MIN_SAMPLES:
        return None, "Yüz yeterince görülemedi. Işığı ve kamerayı kontrol edip tekrar dene."
    med, spread = summarize(samples)
    if spread > MAX_SPREAD:
        return None, "Ölçüm oynak: ölçüm sırasında kıpırdamadan dur ve tekrar dene."
    fm = focal_mult_from_measurement(med, known_z_cm, ipd_cm, frame_w)
    lo, hi = FOCAL_MULT_RANGE
    if not lo <= fm <= hi:
        return None, f"Sonuç ({fm:.2f}) beklenen aralığın dışında. Mesafeyi ve göz mesafesi ayarını kontrol et."
    return fm, "Ölçüm başarılı."


# --------------------------------------------------------------------------- ekran akışı

def _preview(frame, pose, width=480):
    """Kamera karesini göz işaretleriyle küçültüp GL'e basılabilir RGB baytlarına çevirir."""
    img = frame.copy()
    if pose is not None:
        for key in ("left_eye_px", "right_eye_px"):
            cv2.circle(img, pose[key], 5, (0, 255, 0), 2)
        cv2.line(img, pose["left_eye_px"], pose["right_eye_px"], (0, 255, 0), 1)
    h = int(img.shape[0] * width / img.shape[1])
    img = cv2.resize(img, (width, h))
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return np.ascontiguousarray(np.flipud(rgb)).tobytes(), width, h


def run_calibration(menu, tracker, settings, frame_w):
    """
    Kalibrasyon ekranı. Başarılıysa settings.focal_mult güncellenir ve kaydedilir.
    :param tracker: TrackerThread (latest_frame / latest_measurement kullanılır)
    :return: False ise pencere kapatıldı (çıkış istendi), aksi halde True
    """
    renderer = menu.renderer
    distance_idx = DISTANCES_CM.index(60)
    state = "ready"            # ready -> measuring -> done
    samples, last_seq, t_start = [], -1, 0.0
    result_fm, message = None, ""
    clock = pygame.time.Clock()

    while True:
        for e in pygame.event.get():
            if e.type == QUIT:
                return False
            if e.type == VIDEORESIZE:
                renderer.resize((e.w, e.h))
            elif e.type == KEYDOWN:
                if e.key == K_ESCAPE:
                    return True
                if state == "ready":
                    if e.key == K_UP:
                        distance_idx = min(distance_idx + 1, len(DISTANCES_CM) - 1)
                    elif e.key == K_DOWN:
                        distance_idx = max(distance_idx - 1, 0)
                    elif e.key in (K_RETURN, K_KP_ENTER):
                        state, samples, last_seq, t_start = "measuring", [], -1, time.monotonic()
                elif state == "done":
                    if e.key in (K_RETURN, K_KP_ENTER) and result_fm is not None:
                        settings.focal_mult = result_fm
                        settings.clamp()
                        settings.save()
                        return True
                    if e.key == K_r or e.key in (K_RETURN, K_KP_ENTER):
                        state = "ready"

        known_z = DISTANCES_CM[distance_idx]
        frame, pose = tracker.latest_frame()
        _, stamp, seq = tracker.latest_measurement()
        face_now = pose is not None and time.monotonic() - stamp < 0.5

        if state == "measuring":
            if face_now and seq != last_seq:
                last_seq = seq
                samples.append(pose["ipd_px"])
            if time.monotonic() - t_start >= MEASURE_SECONDS:
                result_fm, message = evaluate(samples, known_z, settings.ipd_cm, frame_w)
                state = "done"

        w, h = renderer.size
        glClearColor(0.02, 0.02, 0.06, 1.0)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        _blit_text(menu.title_font, "Kalibrasyon", TITLE_COLOR, 80, 40, h)
        if frame is not None:
            data, pw, ph = _preview(frame, pose if face_now else None)
            _blit_image(data, pw, ph, w - pw - 80, 120, h)

        x, y = 80, 130
        line = lambda text, color=TEXT_COLOR, font=menu.item_font, dy=48: (
            _blit_text(font, text, color, x, line.y, h), setattr(line, "y", line.y + dy))
        line.y = y
        if state == "ready":
            line("Gözlerinin ekrandan uzaklığını ölç:", TEXT_COLOR)
            line(f"Mesafe:  {known_z} cm   (Yukarı/Aşağı ile değiştir)", SELECTED_COLOR)
            line("Bir cetvel ya da kol boyunla ekrandan bu mesafeye otur,", DIM_COLOR, menu.small_font, 32)
            line("kameraya düz bak ve Enter'a bas. Ölçüm sırasında kıpırdama.", DIM_COLOR, menu.small_font, 40)
            if face_now:
                line(f"Şu an tahmin edilen mesafe: {pose['z_cm']:.0f} cm", TEXT_COLOR)
                line(f"Yüz bulundu. Enter: ölç    ESC: iptal", DIM_COLOR, menu.small_font)
            else:
                line("Yüz bulunamadı — kameranın karşısına geç.", (255, 120, 120))
        elif state == "measuring":
            progress = min(1.0, (time.monotonic() - t_start) / MEASURE_SECONDS)
            line("Ölçülüyor — kıpırdamadan dur...", SELECTED_COLOR)
            line(f"{int(progress * 100)}%   ({len(samples)} örnek)", TEXT_COLOR)
        else:
            if result_fm is not None:
                line(message, (120, 230, 140))
                line(f"Odak çarpanı:  {settings.focal_mult:.2f}  ->  {result_fm:.2f}", TEXT_COLOR)
                line("Enter: kaydet    R: tekrar ölç    ESC: vazgeç", DIM_COLOR, menu.small_font)
            else:
                line(message, (255, 120, 120), menu.small_font, 36)
                line("Enter / R: tekrar dene    ESC: vazgeç", DIM_COLOR, menu.small_font)
        pygame.display.flip()
        clock.tick(30)
