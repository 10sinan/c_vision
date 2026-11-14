"""
Ana uygulama - Kafa Takipli Gerçek Zamanlı 3D Parallaks (Aşama 4)

Tuşlar:
  ESC / Q        Çıkış
  F              Tam ekran
  H              HUD (pencere başlığında FPS/poz) aç/kapat
  Sol / Sağ      Odak uzaklığı çarpanı  -/+ 0.05  (derinlik/yan ölçeği)
  Aşağı / Yukarı Kamera ofseti (ekran merkezinden yukarı) -/+ 0.5 cm
  - / +          EMA alpha -/+ 0.05
  PageUp/PageDn  Model boyutu +/- %10 (--model ile)
  L              Oda çizgilerini aç/kapat
  R              Kalibrasyonu varsayılana döndür

Kamerasız test: python main.py --mouse
3B model:       python main.py --model araba.glb   (.obj / .glb / .gltf)
"""

import sys
import pygame
from pygame.locals import (QUIT, KEYDOWN, VIDEORESIZE, K_ESCAPE, K_q, K_f, K_h,
                           K_LEFT, K_RIGHT, K_UP, K_DOWN, K_r, K_MINUS, K_PLUS,
                           K_EQUALS, K_KP_MINUS, K_KP_PLUS, K_l, K_PAGEUP,
                           K_PAGEDOWN)

from head_tracker import HeadTracker
from tracker_thread import TrackerThread
from signal_filter import PoseNormalizer, EMAFilter
from off_axis import head_to_eye_cm, CAMERA_OFFSET_Y_CM
from renderer import Renderer

FRAME_W, FRAME_H = 640, 480
DEFAULT_FOCAL_MULT = 0.85
DEFAULT_ALPHA = 0.8
NEUTRAL_Z_CM = 60.0
FACE_LOST_TIMEOUT_S = 1.0     # bu süreden sonra merkeze dönüş başlar
RETURN_RATE = 0.03            # kare başına merkeze yaklaşma oranı


def main():
    use_mouse = "--mouse" in sys.argv
    model_path = None
    if "--model" in sys.argv:
        i = sys.argv.index("--model")
        if i + 1 >= len(sys.argv):
            sys.exit("Kullanım: python main.py --model dosya.glb")
        model_path = sys.argv[i + 1]
    renderer = Renderer(model_path=model_path)
    clock = pygame.time.Clock()
    ema = EMAFilter(alpha=DEFAULT_ALPHA)
    normalizer = PoseNormalizer(FRAME_W, FRAME_H)

    focal_mult = DEFAULT_FOCAL_MULT
    cam_offset = CAMERA_OFFSET_Y_CM
    hud = True

    tracker = None
    if not use_mouse:
        tracker = TrackerThread(HeadTracker(width=FRAME_W, height=FRAME_H))
        tracker.start()

    eye = (0.0, cam_offset, NEUTRAL_Z_CM)
    running = True
    try:
        while running:
            dt = clock.tick(60) / 1000.0
            for e in pygame.event.get():
                if e.type == QUIT:
                    running = False
                elif e.type == VIDEORESIZE:
                    renderer.resize((e.w, e.h))
                elif e.type == KEYDOWN:
                    k = e.key
                    if k in (K_ESCAPE, K_q): running = False
                    elif k == K_f: pygame.display.toggle_fullscreen()
                    elif k == K_h: hud = not hud
                    elif k == K_RIGHT: focal_mult += 0.05
                    elif k == K_LEFT: focal_mult = max(0.2, focal_mult - 0.05)
                    elif k == K_UP: cam_offset += 0.5
                    elif k == K_DOWN: cam_offset -= 0.5
                    elif k in (K_PLUS, K_EQUALS, K_KP_PLUS):
                        ema.alpha = min(1.0, ema.alpha + 0.05)
                    elif k in (K_MINUS, K_KP_MINUS):
                        ema.alpha = max(0.05, ema.alpha - 0.05)
                    elif k == K_l: renderer.show_room = not renderer.show_room
                    elif k == K_PAGEUP: renderer.model_scale *= 1.1
                    elif k == K_PAGEDOWN: renderer.model_scale /= 1.1
                    elif k == K_r:
                        focal_mult, cam_offset = DEFAULT_FOCAL_MULT, CAMERA_OFFSET_Y_CM
                        ema.alpha = DEFAULT_ALPHA
                        renderer.model_scale = 1.0

            focal_px = FRAME_W * focal_mult
            status = "takip"

            if use_mouse:
                mx, my = pygame.mouse.get_pos()
                w, h = renderer.size
                raw = ((mx / w) * 2 - 1, 1 - (my / h) * 2, NEUTRAL_Z_CM)
                ema.update(raw)
            else:
                if tracker.failed:
                    break
                # Aynı odak uzaklığı hem Z (derinlik) hem X/Y dönüşümünde kullanılır
                tracker.tracker.focal_length_px = focal_px
                pose, age = tracker.latest()
                if pose is not None and age < FACE_LOST_TIMEOUT_S:
                    ema.update(normalizer.normalize(pose))
                else:
                    status = "yuz yok"
                    if ema.state is not None:
                        ema.state = tuple(s + RETURN_RATE * (n - s) for s, n in
                                          zip(ema.state, (0.0, 0.0, NEUTRAL_Z_CM)))

            if ema.state is not None:
                x_n, y_n, z = ema.state
                eye = head_to_eye_cm(x_n, y_n, z, FRAME_W, FRAME_H, focal_px, cam_offset)

            renderer.draw(eye, dt)

            if hud:
                pygame.display.set_caption(
                    f"Parallaks | {clock.get_fps():.0f} FPS | {status} | "
                    f"Z={eye[2]:.0f}cm | f={focal_mult:.2f} off={cam_offset:.1f} a={ema.alpha:.2f}")
    finally:
        if tracker:
            tracker.stop()
        pygame.quit()


if __name__ == "__main__":
    main()
