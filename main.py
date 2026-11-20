"""
Ana uygulama - Kafa Takipli Gerçek Zamanlı 3D Parallaks (Aşama 4)

Tuşlar:
  ESC / Q        Çıkış
  F              Tam ekran
  H              HUD (pencere başlığında FPS/poz) aç/kapat
  Tab            Ayar panelini aç/kapat (sağ üstteki "Ayarlar" düğmesi de açar; aşağıdaki
                 tuşların hepsi panelden fareyle de yapılabilir)
  Sol / Sağ      Odak uzaklığı çarpanı  -/+ 0.05  (derinlik/yan ölçeği)
  Aşağı / Yukarı Kamera ofseti (ekran merkezinden yukarı) -/+ 0.5 cm
  - / +          Titreme yumuşatma: daha pürüzsüz / daha hızlı tepki
  , / .          Hareket gücü -/+ 0.25 (1 = gerçek ölçü; büyüdükçe baş hareketi daha çok etki eder)
  R              Kalibrasyonu (ve model görünümünü) varsayılana döndür

  M              Sahne seçim menüsüne dön
  P              Ayarlar (ekran boyutu, kamera konumu, göz mesafesi)
  K              Mesafe kalibrasyonu (kamera varken)
  J              Gölgeleri aç/kapa (ekran kartı destekliyorsa)
  O              Işık/sis görünümü: Doğal / Yumuşak / Dramatik (Shift+O: geriye)
  E              Efektleri aç/kapa
  N              Efekt türü: Yaprak / Yağmur / Kar / Ateşböceği (Shift+N: geriye)
  [ / ]          Efekt yoğunluğu -/+ 0.25
  V              Video kaydı aç/kapa (sahne + kamera yan yana, kayitlar/ klasörüne)

Açılışta proje klasöründeki modeller (.glb/.gltf/.obj) listelenir ve seçim istenir.
--model verilirse menü atlanır.

Model tuşları (model açıkken; ayarlar çıkışta <model>.view.json'a kaydedilir):
  PageUp/PageDn  Boyut +/- %10
  W / S          Sahneyi yaklaştır / uzaklaştır (2 cm)
  A / D          Sahneyi sola / sağa döndür (5°)
  T / G          Sahneyi üstten / alttan bakışa eğ (5°)
  U              Oda: model, ekranın arkasındaki duvarlı kutu odanın içinde durur / oda yok
  C              Ekranı doldur / ekrana sığdır
  B              Öne çıkar: sahnenin ortası ekran düzleminde (baş oynayınca yerinde durur, etrafı
                 görünür) / tümü camın arkasında (pencereden bakış)
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
                           K_PAGEDOWN, K_w, K_s, K_a, K_d, K_t, K_g, K_m, K_p, K_k, K_v, K_e, K_LEFTBRACKET, K_RIGHTBRACKET, K_o, K_j,
                           K_TAB, K_n, K_b, K_u, K_COMMA, K_PERIOD, KMOD_SHIFT, MOUSEBUTTONDOWN, MOUSEWHEEL)

from head_tracker import HeadTracker
from tracker_thread import TrackerThread
import time

from signal_filter import PoseNormalizer, OneEuroFilter, OutlierGate, decay_toward, blend
from off_axis import head_to_eye_cm, amplify_eye
from settings import Settings, nearest_aspect_index, ASPECT_PRESETS
from renderer import Renderer, LOOKS
from effects import KIND_NAMES
from menu import Menu, find_models, NO_MODEL, SETTINGS_ITEM, CALIBRATION_ITEM
from calibration import run_calibration
from recorder import Recorder
from panel import Panel, Row

FRAME_W, FRAME_H = 640, 480
# One Euro filtre ayarları, bileşen başına (x_n, y_n, z_cm).
# Ölçüm: Z (IPD'den) kare başına ~1.2 cm gürültülü, X/Y ise <1 piksel -> Z daha güçlü filtrelenir.
FILTER_MIN_CUTOFF = (1.0, 1.0, 0.3)   # Hz; düşük = daha pürüzsüz, daha gecikmeli
FILTER_BETA = (3.0, 3.0, 0.05)        # hıza tepki; yüksek = hızlı harekette daha az gecikme
# Hız tahmini filtresi (Hz). Simülasyon (0.7 Hz baş sallama, 30 Hz kamera): 1 -> 3 Hz hareket
# hatasını ~1/3 azaltır; durgun titreme artar ama ~0.2 kamera pikselinin altında kalır.
FILTER_D_CUTOFF = 3.0
PREDICT_HORIZON_S = 0.05              # ölçümler arası hareket tahmini en fazla bu kadar ileri
PREDICT_FADE_S = 0.15                 # ölçüm gecikirse tahmin bu zaman sabitiyle söner
PREDICT_AXES = (0, 1)                 # yalnızca X/Y tahmin edilir; Z gürültülü olduğu için hariç
NEUTRAL_Z_CM = 60.0
FACE_LOST_TIMEOUT_S = 1.0     # bu süreden sonra merkeze dönüş başlar
RETURN_TAU_S = 0.55           # merkeze dönüş zaman sabiti (kare hızından bağımsız)
REACQUIRE_BLEND_S = 0.25      # yüz geri gelince eski konumdan yenisine yumuşak geçiş süresi


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
    menu_items = models + [NO_MODEL] + ([] if use_mouse else [CALIBRATION_ITEM]) + [SETTINGS_ITEM]

    # Kamera menüden ÖNCE başlar: ısınır ve kalibrasyon ekranı hemen canlı görüntü verir.
    tracker = None
    if not use_mouse:
        tracker = TrackerThread(HeadTracker(width=FRAME_W, height=FRAME_H, real_ipd_cm=settings.ipd_cm))
        tracker.tracker.focal_length_px = FRAME_W * settings.focal_mult
        tracker.start()

    def calibrate():
        """Kalibrasyon ekranı; sonucu takipçiye uygular. :return: False ise çıkış istendi."""
        ok = run_calibration(menu, tracker, settings, FRAME_W)
        tracker.tracker.focal_length_px = FRAME_W * settings.focal_mult
        return ok

    def quit_early():
        if tracker:
            tracker.stop()
        pygame.quit()

    def open_settings():
        """Ayarlar ekranını açar, kaydeder ve uygular. :return: False ise çıkış istendi."""
        ok = menu.edit_settings(settings)
        settings.save()
        renderer.set_screen(*settings.screen_cm())
        if tracker:
            tracker.tracker.real_ipd_cm = settings.ipd_cm
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
            if choice == CALIBRATION_ITEM:
                if not calibrate():
                    return False
                start = menu_items.index(CALIBRATION_ITEM)
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
        if not settings_existed and not (open_settings() and menu.show_guide(tracker is not None)):
            quit_early()
            return
        if models:
            menu.prepare_previews(menu_items)
            if not pick_and_load(None):
                quit_early()
                return
    clock = pygame.time.Clock()
    filt = OneEuroFilter(FILTER_MIN_CUTOFF, FILTER_BETA, d_cutoff=FILTER_D_CUTOFF)
    gate = OutlierGate()
    smooth = 1.0          # +/- ile değişen yumuşatma çarpanı (min_cutoff'a bölünür)
    last_seq = -1
    last_ok_t = float("-inf")   # son kabul edilen ölçümün zamanı
    lost = True                 # yüz kayıp mı (merkeze dönüş sürüyor)
    blend_from, blend_t0 = None, 0.0   # yüz geri gelince yumuşak geçişin başlangıcı
    lat_ms, meas_hz = 0.0, 0.0  # HUD: ölçüm yaşı (yakalama -> filtre) ve saniyedeki ölçüm sayısı
    meas_count, meas_t0 = 0, time.monotonic()
    normalizer = PoseNormalizer(FRAME_W, FRAME_H)

    focal_mult = settings.focal_mult
    cam_offset = settings.cam_offset_y_cm()
    gain = settings.parallax_gain    # baş hareketi çarpanı (çıkışta settings.json'a yazılır)
    hud = True

    recorder = Recorder()
    last_saved = None     # son kaydedilen video yolu (başlıkta gösterilir)
    eye = (0.0, cam_offset, NEUTRAL_Z_CM)

    # Ayar paneli: her satır ilgili klavye tuşunu "basar", böylece tuşlarla tek bir kod yolu paylaşılır.
    def press(plus, minus=None, minus_mod=0):
        def change(direction):
            key, mod = (plus, 0) if direction > 0 or minus is None else (minus, minus_mod)
            pygame.event.post(pygame.event.Event(KEYDOWN, key=key, mod=mod, unicode=""))
        return change

    has_model = lambda: renderer.model_list is not None
    panel = Panel([
        Row("Işık görünümü", "step", press(K_o, K_o, KMOD_SHIFT), lambda: LOOKS[renderer.look]["name"], has_model),
        Row("Gölgeler", "toggle", press(K_j), lambda: renderer.shadows,
            lambda: has_model() and renderer.sh.available),
        Row("Efektler", "toggle", press(K_e), lambda: renderer.effects, has_model),
        Row("Efekt türü", "step", press(K_n, K_n, KMOD_SHIFT), lambda: KIND_NAMES[renderer.fx_kind], has_model),
        Row("Efekt yoğunluğu", "step", press(K_RIGHTBRACKET, K_LEFTBRACKET),
            lambda: f"{renderer.fx_density:.2f}", has_model),
        Row("Oda (duvarlı kutu)", "toggle", press(K_u), lambda: renderer.room, has_model),
        Row("Öne çıkar (etrafına bak)", "toggle", press(K_b), lambda: renderer.popout,
            lambda: has_model() and not renderer.room),
        Row("Ekranı doldur", "toggle", press(K_c), lambda: renderer.cover,
            lambda: has_model() and not renderer.room),
        Row("Boyut", "step", press(K_PAGEUP, K_PAGEDOWN), lambda: f"%{renderer.model_scale * 100:.0f}", has_model),
        Row("Yakınlık", "step", press(K_w, K_s), lambda: f"{renderer.depth:+.0f} cm", has_model),
        Row("Döndürme", "step", press(K_d, K_a), lambda: f"{renderer.yaw:+.0f}°", has_model),
        Row("Eğim", "step", press(K_t, K_g), lambda: f"{renderer.pitch:+.0f}°", has_model),
        Row("Oda çizgileri", "toggle", press(K_l), lambda: renderer.show_room),
        Row("Hareket gücü", "step", press(K_PERIOD, K_COMMA), lambda: f"{gain:.2f} kat"),
        Row("Yumuşatma", "step", press(K_MINUS, K_PLUS), lambda: f"{smooth:.2f}"),
        Row("Odak çarpanı", "step", press(K_RIGHT, K_LEFT), lambda: f"{focal_mult:.2f}"),
        Row("Kamera yüksekliği", "step", press(K_UP, K_DOWN), lambda: f"{cam_offset:.1f} cm"),
        Row("Video kaydı", "toggle", press(K_v), lambda: recorder.active),
        Row("Sahne menüsü", "action", press(K_m)),
        Row("Ekran ve göz ayarları", "action", press(K_p)),
    ] + ([Row("Mesafe kalibrasyonu", "action", press(K_k))] if tracker else []) + [
        Row("Varsayılana dön", "action", press(K_r)),
    ])
    running = True
    try:
        while running:
            dt = clock.tick(60) / 1000.0
            for e in pygame.event.get():
                if e.type == QUIT:
                    running = False
                elif e.type == VIDEORESIZE:
                    renderer.resize((e.w, e.h))
                elif e.type == MOUSEBUTTONDOWN and e.button == 1:
                    panel.click(e.pos, renderer.size)
                elif e.type == MOUSEWHEEL:
                    panel.wheel(pygame.mouse.get_pos(), renderer.size, e.y)
                elif e.type == KEYDOWN:
                    k = e.key
                    if k in (K_ESCAPE, K_q): running = False
                    elif k == K_TAB: panel.open = not panel.open
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
                    elif k == K_PERIOD: gain = min(6.0, gain + 0.25)
                    elif k == K_COMMA: gain = max(0.5, gain - 0.25)
                    elif k == K_p:
                        if not open_settings():
                            running = False
                        cam_offset = settings.cam_offset_y_cm()
                        if tracker:
                            tracker.tracker.real_ipd_cm = settings.ipd_cm
                        clock.tick()
                    elif k == K_e: renderer.effects = not renderer.effects
                    elif k == K_j: renderer.shadows = not renderer.shadows
                    elif k == K_n:
                        renderer.fx_kind = (renderer.fx_kind + (-1 if e.mod & KMOD_SHIFT else 1)) % len(KIND_NAMES)
                        renderer.sync_effects()
                    elif k == K_o: renderer.look = (renderer.look + (-1 if e.mod & KMOD_SHIFT else 1)) % len(LOOKS)
                    elif k in (K_LEFTBRACKET, K_RIGHTBRACKET):
                        renderer.fx_density = min(3.0, max(0.0, renderer.fx_density + (0.25 if k == K_RIGHTBRACKET else -0.25)))
                        renderer.sync_effects()
                    elif k == K_v:
                        if recorder.active:
                            last_saved = recorder.stop()
                            print(f"[KAYIT] Kaydedildi: {last_saved}  "
                                  f"({recorder.written / 30:.1f} sn, {recorder.dropped} kare atlandı)")
                        else:
                            print("[KAYIT] Başladı:", recorder.start())
                            last_saved = None
                    elif k == K_k and tracker:
                        if not calibrate():
                            running = False
                        focal_mult = settings.focal_mult
                        clock.tick()
                    elif k == K_m:
                        if not pick_and_load(renderer.model_path):
                            running = False
                        clock.tick()   # menüde geçen süre dt'ye eklenmesin
                    elif k == K_l: renderer.show_room = not renderer.show_room
                    elif k == K_c: renderer.cover = not renderer.cover
                    elif k == K_b: renderer.popout = not renderer.popout
                    elif k == K_u: renderer.room = not renderer.room
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
                        gain = Settings().parallax_gain
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
                # Filtre yalnızca yeni kamera ölçümüyle (~30 Hz) ve ölçüm zamanıyla güncellenir;
                # aynı pozu her render karesinde tekrar beslemek titremeye yol açar.
                if pose is not None and seq != last_seq:
                    last_seq = seq
                    meas = normalizer.normalize(pose)
                    if gate.accept(meas, stamp):   # tek karelik sıçramalar elenir
                        if lost:
                            # Yüz geri geldi: eski hız/zaman durumu atılır, görüntü zıplamasın
                            # diye merkeze dönmüş konumdan yeni konuma yumuşak geçilir.
                            lost = False
                            blend_from = filt.state or (0.0, 0.0, NEUTRAL_Z_CM)
                            blend_t0 = now
                            filt.reset()
                        filt.update(meas, stamp)
                        last_ok_t = stamp
                        lat_ms += 0.1 * ((now - stamp) * 1000.0 - lat_ms)
                        meas_count += 1
                if now - meas_t0 >= 1.0:
                    meas_hz, meas_count, meas_t0 = meas_count / (now - meas_t0), 0, now
                if now - last_ok_t >= FACE_LOST_TIMEOUT_S:
                    status = "yuz yok"
                    if not lost:
                        lost = True
                        gate.reset()
                    if filt.state is not None:
                        filt.set_state(decay_toward(filt.state, (0.0, 0.0, NEUTRAL_Z_CM),
                                                    dt, RETURN_TAU_S))

            if filt.state is not None:
                if use_mouse or status != "takip":
                    x_n, y_n, z = filt.state
                else:
                    now = time.monotonic()
                    x_n, y_n, z = filt.predict(now, PREDICT_HORIZON_S, PREDICT_AXES,
                                               fade=PREDICT_FADE_S)
                    if blend_from is not None:
                        k = (now - blend_t0) / REACQUIRE_BLEND_S
                        if k >= 1.0:
                            blend_from = None
                        else:
                            x_n, y_n, z = blend(blend_from, (x_n, y_n, z), k)
                eye = amplify_eye(head_to_eye_cm(x_n, y_n, z, FRAME_W, FRAME_H, focal_px, cam_offset),
                                  1.0 if use_mouse else gain, cam_offset)   # fare zaten tüm aralığı gezer

            renderer.draw(eye, dt, present=False)
            if recorder.active:
                # Kare, ekrana basılmadan ve panel çizilmeden ÖNCE okunur (flip sonrası arka tampon
                # tanımsız; panel videoya girmesin)
                if recorder.due():
                    cam_frame, cam_pose = tracker.latest_frame() if tracker else (None, None)
                    if cam_pose is not None and time.monotonic() - tracker.latest_measurement()[1] > 0.5:
                        cam_pose = None   # yüz kaybolmuş: eski göz işaretlerini gösterme
                    recorder.capture(renderer.grab_frame(), cam_frame, cam_pose, eye[2])
            panel.draw(renderer.size, pygame.mouse.get_pos())
            pygame.display.flip()

            if hud:
                rec = f" | REC {recorder.elapsed():.0f}s" if recorder.active else (
                    f" | kaydedildi: {os.path.basename(last_saved)}" if last_saved else "")
                trk = "" if use_mouse else f"gec={lat_ms:.0f}ms olc={meas_hz:.0f}Hz | "
                pygame.display.set_caption(
                    f"Parallaks | {clock.get_fps():.0f} FPS | {status} | {trk}"
                    f"Z={eye[2]:.0f}cm | f={focal_mult:.2f} off={cam_offset:.1f} yum={smooth:.2f}{rec}")
    finally:
        if recorder.active:
            print("[KAYIT] Kaydedildi:", recorder.stop())
        renderer.save_view()
        settings.parallax_gain = gain
        settings.save()
        if tracker:
            tracker.stop()
        pygame.quit()


if __name__ == "__main__":
    main()
