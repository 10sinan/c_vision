"""
Ana uygulama - Kafa Takipli Gerçek Zamanlı 3D Parallaks (Aşama 4)

Tuşlar:
  ESC / Q        Çıkış
  F              Tam ekran
  H              HUD (pencere başlığında FPS/poz) aç/kapat
  Sol / Sağ      Odak uzaklığı çarpanı  -/+ 0.05  (derinlik/yan ölçeği)
  Aşağı / Yukarı Kamera ofseti (ekran merkezinden yukarı) -/+ 0.5 cm
  - / +          Titreme yumuşatma: daha pürüzsüz / daha hızlı tepki
  R              Kalibrasyonu (ve model görünümünü) varsayılana döndür

  M              Sahne seçim menüsüne dön
  P              Ayarlar (ekran boyutu, kamera konumu, göz mesafesi)

Açılışta proje klasöründeki modeller (.glb/.gltf/.obj) listelenir ve seçim istenir.
--model verilirse menü atlanır.

Model tuşları (model açıkken; ayarlar çıkışta <model>.view.json'a kaydedilir):
  PageUp/PageDn  Boyut +/- %10
  W / S          Sahneyi yaklaştır / uzaklaştır (2 cm)
  A / D          Sahneyi sola / sağa döndür (5°)
  T / G          Sahneyi üstten / alttan bakışa eğ (5°)
  C              Ekranı doldur / ekrana sığdır
  L              Oda çizgilerini aç/kapat

Kamerasız test: python main.py --mouse
3B model:       python main.py --model araba.glb   (.obj / .glb / .gltf)
"""

import os
import sys
import pygame
from pygame.locals import (QUIT, KEYDOWN, VIDEORESIZE, K_ESCAPE, K_q, K_f, K_h,
                           K_LEFT, K_RIGHT, K_UP, K_DOWN, K_r, K_MINUS, K_PLUS,
                           K_EQUALS, K_KP_MINUS, K_KP_PLUS, K_l, K_c, K_PAGEUP,
                           K_PAGEDOWN, K_w, K_s, K_a, K_d, K_t, K_g, K_m, K_p)

from head_tracker import HeadTracker
from tracker_thread import TrackerThread
import time

from signal_filter import PoseNormalizer, OneEuroFilter
from off_axis import head_to_eye_cm
from settings import Settings, nearest_aspect_index, ASPECT_PRESETS
from renderer import Renderer
from menu import Menu, find_models, NO_MODEL, SETTINGS_ITEM

FRAME_W, FRAME_H = 640, 480
# One Euro filtre ayarları, bileşen başına (x_n, y_n, z_cm).
# Ölçüm: Z (IPD'den) kare başına ~1.2 cm gürültülü, X/Y ise <1 piksel -> Z daha güçlü filtrelenir.
FILTER_MIN_CUTOFF = (1.0, 1.0, 0.3)   # Hz; düşük = daha pürüzsüz, daha gecikmeli
FILTER_BETA = (3.0, 3.0, 0.05)        # hıza tepki; yüksek = hızlı harekette daha az gecikme
PREDICT_HORIZON_S = 0.05              # ölçümler arası hareket tahmini en fazla bu kadar ileri
PREDICT_AXES = (0, 1)                 # yalnızca X/Y tahmin edilir; Z gürültülü olduğu için hariç
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
    renderer = Renderer()
    menu = Menu(renderer)
    try:   # ayar dosyası yokken en-boy oranı için masaüstü çözünürlüğünü kullan
        dw, dh = pygame.display.get_desktop_sizes()[0]
        default_aspect = ASPECT_PRESETS[nearest_aspect_index(dw / dh)][1]
    except Exception:
        default_aspect = 16 / 9
    settings, settings_existed = Settings.load(default_aspect=default_aspect)
    renderer.set_screen(*settings.screen_cm())
    models = find_models(os.path.dirname(os.path.abspath(__file__)))
    menu_items = models + [NO_MODEL, SETTINGS_ITEM]

    def open_settings():
        """Ayarlar ekranını açar, kaydeder ve uygular. :return: False ise çıkış istendi."""
        ok = menu.edit_settings(settings)
        settings.save()
        renderer.set_screen(*settings.screen_cm())
        return ok

    def pick_and_load(current):
        """Menüyü gösterir, seçilen modeli yükler. :return: False ise çıkış istendi."""
        start = menu_items.index(current) if current in menu_items else 0
        while True:
            choice = menu.choose(menu_items, start)
            if choice is False:
                return False
            if choice == SETTINGS_ITEM:
                if not open_settings():
                    return False
                start = menu_items.index(SETTINGS_ITEM)
                continue
            break
        if choice is not NO_MODEL:
            menu.show_loading(choice)
        renderer.load_model(choice)
        return True

    if model_path is not None:
        renderer.load_model(model_path)
    else:
        # İlk açılışta (ayar dosyası yok) ekran boyutunu soruyoruz
        if not settings_existed and not open_settings():
            pygame.quit()
            return
        if models and not pick_and_load(None):
            pygame.quit()
            return
    clock = pygame.time.Clock()
    filt = OneEuroFilter(FILTER_MIN_CUTOFF, FILTER_BETA)
    smooth = 1.0          # +/- ile değişen yumuşatma çarpanı (min_cutoff'a bölünür)
    last_seq = -1
    normalizer = PoseNormalizer(FRAME_W, FRAME_H)

    focal_mult = settings.focal_mult
    cam_offset = settings.cam_offset_y_cm()
    hud = True

    tracker = None
    if not use_mouse:
        tracker = TrackerThread(HeadTracker(width=FRAME_W, height=FRAME_H, real_ipd_cm=settings.ipd_cm))
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
                        smooth = max(0.25, smooth / 1.25)   # daha hızlı tepki
                    elif k in (K_MINUS, K_KP_MINUS):
                        smooth = min(8.0, smooth * 1.25)    # daha pürüzsüz
                    elif k == K_p:
                        if not open_settings():
                            running = False
                        cam_offset = settings.cam_offset_y_cm()
                        if tracker:
                            tracker.tracker.real_ipd_cm = settings.ipd_cm
                        clock.tick()
                    elif k == K_m:
                        if not pick_and_load(renderer.model_path):
                            running = False
                        clock.tick()   # menüde geçen süre dt'ye eklenmesin
                    elif k == K_l: renderer.show_room = not renderer.show_room
                    elif k == K_c: renderer.cover = not renderer.cover
                    elif k == K_w: renderer.depth += 2.0
                    elif k == K_s: renderer.depth -= 2.0
                    elif k == K_a: renderer.yaw -= 5.0
                    elif k == K_d: renderer.yaw += 5.0
                    elif k == K_t: renderer.pitch += 5.0
                    elif k == K_g: renderer.pitch -= 5.0
                    elif k == K_PAGEUP: renderer.model_scale *= 1.1
                    elif k == K_PAGEDOWN: renderer.model_scale /= 1.1
                    elif k == K_r:
                        focal_mult, cam_offset = settings.focal_mult, settings.cam_offset_y_cm()
                        smooth = 1.0
                        renderer.reset_view()

            focal_px = FRAME_W * focal_mult
            status = "takip"
            filt.min_cutoff = [c / smooth for c in FILTER_MIN_CUTOFF]

            if use_mouse:
                mx, my = pygame.mouse.get_pos()
                w, h = renderer.size
                raw = ((mx / w) * 2 - 1, 1 - (my / h) * 2, NEUTRAL_Z_CM)
                filt.update(raw, time.monotonic())
            else:
                if tracker.failed:
                    break
                # Aynı odak uzaklığı hem Z (derinlik) hem X/Y dönüşümünde kullanılır
                tracker.tracker.focal_length_px = focal_px
                pose, stamp, seq = tracker.latest_measurement()
                now = time.monotonic()
                if pose is not None and now - stamp < FACE_LOST_TIMEOUT_S:
                    # Filtre yalnızca yeni kamera ölçümüyle (~30 Hz) ve ölçüm zamanıyla güncellenir;
                    # aynı pozu her render karesinde tekrar beslemek titremeye yol açar.
                    if seq != last_seq:
                        last_seq = seq
                        filt.update(normalizer.normalize(pose), stamp)
                else:
                    status = "yuz yok"
                    if filt.state is not None:
                        filt.set_state(tuple(s + RETURN_RATE * (n - s) for s, n in
                                             zip(filt.state, (0.0, 0.0, NEUTRAL_Z_CM))))

            if filt.state is not None:
                if use_mouse or status != "takip":
                    x_n, y_n, z = filt.state
                else:
                    x_n, y_n, z = filt.predict(time.monotonic(), PREDICT_HORIZON_S, PREDICT_AXES)
                eye = head_to_eye_cm(x_n, y_n, z, FRAME_W, FRAME_H, focal_px, cam_offset)

            renderer.draw(eye, dt)

            if hud:
                pygame.display.set_caption(
                    f"Parallaks | {clock.get_fps():.0f} FPS | {status} | "
                    f"Z={eye[2]:.0f}cm | f={focal_mult:.2f} off={cam_offset:.1f} yum={smooth:.2f}")
    finally:
        renderer.save_view()
        if tracker:
            tracker.stop()
        pygame.quit()


if __name__ == "__main__":
    main()
