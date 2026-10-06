"""
Aşama 3 Demo - Kafa takipli off-axis parallaks.
Tuşlar: ESC/Q çıkış, F tam ekran (illüzyon için önerilir).
Fare ile test (kamera yoksa): --mouse
"""

import sys
import pygame
from pygame.locals import QUIT, KEYDOWN, K_ESCAPE, K_q, K_f, VIDEORESIZE

from head_tracker import HeadTracker
from signal_filter import PoseNormalizer, EMAFilter
from off_axis import head_to_eye_cm
from renderer import Renderer

FRAME_W, FRAME_H = 640, 480


def main():
    use_mouse = "--mouse" in sys.argv
    renderer = Renderer()
    clock = pygame.time.Clock()
    ema = EMAFilter(alpha=0.8)
    normalizer = PoseNormalizer(FRAME_W, FRAME_H)
    tracker = None
    if not use_mouse:
        tracker = HeadTracker(width=FRAME_W, height=FRAME_H)
        tracker.start_camera()

    eye = (0.0, 10.0, 60.0)
    running = True
    try:
        while running:
            dt = clock.tick(60) / 1000.0
            for e in pygame.event.get():
                if e.type == QUIT or (e.type == KEYDOWN and e.key in (K_ESCAPE, K_q)):
                    running = False
                elif e.type == KEYDOWN and e.key == K_f:
                    pygame.display.toggle_fullscreen()
                elif e.type == VIDEORESIZE:
                    renderer.resize((e.w, e.h))

            if use_mouse:
                mx, my = pygame.mouse.get_pos()
                w, h = renderer.size
                raw = ((mx / w) * 2 - 1, 1 - (my / h) * 2, 60.0)
                x_n, y_n, z = ema.update(raw)
                eye = head_to_eye_cm(x_n, y_n, z, FRAME_W, FRAME_H, tracker_f())
            else:
                ok, frame, pose = tracker.process_frame()
                if not ok:
                    break
                if pose is not None:
                    x_n, y_n, z = ema.update(normalizer.normalize(pose))
                    eye = head_to_eye_cm(x_n, y_n, z, FRAME_W, FRAME_H,
                                         tracker.focal_length_px)
            renderer.draw(eye, dt)
    finally:
        if tracker:
            tracker.release()
        pygame.quit()


def tracker_f():
    return FRAME_W * 0.85   # HeadTracker ile aynı odak uzaklığı varsayımı


if __name__ == "__main__":
    main()
