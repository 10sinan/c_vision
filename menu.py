"""
Açılış menüsü - proje klasöründeki 3B modelleri listeler, kullanıcı birini seçer.
OpenGL penceresinde çalışır: metin pygame.font ile çizilip glDrawPixels ile basılır.

Kullanım: ↑/↓ veya fare ile seç, Enter / tık ile aç, 1-9 doğrudan seçim, ESC çıkış.
"""

import glob
import os

import pygame
from pygame.locals import (QUIT, KEYDOWN, MOUSEMOTION, MOUSEBUTTONDOWN, VIDEORESIZE,
                           K_ESCAPE, K_q, K_UP, K_DOWN, K_LEFT, K_RIGHT, K_RETURN, K_KP_ENTER,
                           K_SPACE, K_f, KMOD_SHIFT)

from settings import ASPECT_PRESETS, nearest_aspect_index
from OpenGL.GL import (glClear, glClearColor, glWindowPos2d, glDrawPixels, glEnable,
                       glBlendFunc, GL_COLOR_BUFFER_BIT, GL_DEPTH_BUFFER_BIT, GL_RGBA, GL_RGB,
                       GL_UNSIGNED_BYTE, GL_BLEND, GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

MODEL_PATTERNS = ("*.glb", "*.gltf", "*.obj")
NO_MODEL = None  # menüde "model yok" (tel kafes oda) seçeneği
SETTINGS_ITEM = "<ayarlar>"  # menüde ayarlar ekranını açan öğe
CALIBRATION_ITEM = "<kalibrasyon>"  # menüde kalibrasyon ekranını açan öğe (kamera varken)

TITLE_COLOR = (120, 220, 255)
TEXT_COLOR = (215, 215, 225)
DIM_COLOR = (120, 120, 140)
SELECTED_COLOR = (255, 200, 60)


def find_models(folder):
    """:return: klasördeki model dosyalarının sıralı listesi"""
    found = []
    for pattern in MODEL_PATTERNS:
        found += glob.glob(os.path.join(folder, pattern))
    return sorted(found, key=lambda p: os.path.basename(p).lower())


def pretty_name(path):
    """'jungle_environment.glb' -> 'Jungle Environment'"""
    base = os.path.splitext(os.path.basename(path))[0]
    return " ".join(w.capitalize() for w in base.replace("-", " ").replace("_", " ").split())


def _label(path):
    if path is NO_MODEL:
        return "Tel kafes oda (model yok)", ""
    if path == CALIBRATION_ITEM:
        return "Kalibrasyon", "mesafe tahminini yüzüne göre ayarla"
    if path == SETTINGS_ITEM:
        return "Ayarlar", "ekran boyutu, kamera konumu"
    size_mb = os.path.getsize(path) / (1024 * 1024)
    return pretty_name(path), f"{os.path.basename(path)}  ·  {size_mb:.0f} MB"


def _blit_text(font, text, color, x, y_top, win_h):
    """Metni pencerenin (x, y_top) noktasına (sol üst köşe referanslı) çizer."""
    surf = font.render(text, True, color)
    data = pygame.image.tobytes(surf, "RGBA", True)   # OpenGL alttan başlar -> dikey çevrik
    glWindowPos2d(x, win_h - y_top - surf.get_height())
    glDrawPixels(surf.get_width(), surf.get_height(), GL_RGBA, GL_UNSIGNED_BYTE, data)
    return surf.get_width(), surf.get_height()


def _blit_image(rgb_bytes, width, height, x, y_top, win_h):
    """Ham RGB piksel verisini (üstten aşağı satırlar) pencerede (x, y_top) noktasına basar."""
    glWindowPos2d(x, win_h - y_top - height)
    glDrawPixels(width, height, GL_RGB, GL_UNSIGNED_BYTE, rgb_bytes)


class Menu:
    def __init__(self, renderer):
        self.renderer = renderer
        self.title_font = pygame.font.Font(None, 64)
        self.item_font = pygame.font.Font(None, 40)
        self.small_font = pygame.font.Font(None, 26)

    def _layout(self, n):
        """:return: her öğenin (y_üst, yükseklik) satırı"""
        top, row_h = 150, 62
        return [(top + i * row_h, row_h) for i in range(n)]

    def _draw(self, items, selected):
        w, h = self.renderer.size
        glClearColor(0.02, 0.02, 0.06, 1.0)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        x = 80
        _blit_text(self.title_font, "Hangi sahneyi açalım?", TITLE_COLOR, x, 50, h)
        for i, ((y, _), item) in enumerate(zip(self._layout(len(items)), items)):
            name, detail = _label(item)
            sel = i == selected
            prefix = ">  " if sel else "   "
            tw, _ = _blit_text(self.item_font, f"{prefix}{i + 1}.  {name}",
                               SELECTED_COLOR if sel else TEXT_COLOR, x, y, h)
            if detail:
                _blit_text(self.small_font, detail, DIM_COLOR, x + tw + 24, y + 8, h)
        _blit_text(self.small_font,
                   "Yukarı/Aşağı veya fare: seç    Enter / tık: aç    1-9: doğrudan    F: tam ekran    ESC: çıkış",
                   DIM_COLOR, x, h - 60, h)
        pygame.display.flip()

    def _item_at(self, items, pos):
        mx, my = pos
        for i, (y, row_h) in enumerate(self._layout(len(items))):
            if y - 8 <= my < y - 8 + row_h and mx >= 40:
                return i
        return None

    def choose(self, items, selected=0):
        """
        Menüyü gösterir ve kullanıcı seçene kadar bekler.
        :return: seçilen öğe (dosya yolu veya NO_MODEL) ya da çıkış istendiyse False
        """
        clock = pygame.time.Clock()
        while True:
            for e in pygame.event.get():
                if e.type == QUIT:
                    return False
                if e.type == VIDEORESIZE:
                    self.renderer.resize((e.w, e.h))
                elif e.type == KEYDOWN:
                    if e.key in (K_ESCAPE, K_q):
                        return False
                    if e.key == K_f:
                        pygame.display.toggle_fullscreen()
                    elif e.key == K_UP:
                        selected = (selected - 1) % len(items)
                    elif e.key == K_DOWN:
                        selected = (selected + 1) % len(items)
                    elif e.key in (K_RETURN, K_KP_ENTER, K_SPACE):
                        return items[selected]
                    elif e.unicode.isdigit() and 1 <= int(e.unicode) <= min(9, len(items)):
                        return items[int(e.unicode) - 1]
                elif e.type == MOUSEMOTION:
                    i = self._item_at(items, e.pos)
                    if i is not None:
                        selected = i
                elif e.type == MOUSEBUTTONDOWN and e.button == 1:
                    i = self._item_at(items, e.pos)
                    if i is not None:
                        return items[i]
            self._draw(items, selected)
            clock.tick(30)

    def edit_settings(self, settings):
        """
        Ayarlar ekranı: ↑/↓ alan seç, ←/→ değiştir (Shift: büyük adım), Enter/ESC bitir.
        settings (Settings) yerinde değiştirilir.
        :return: False ise pencere kapatıldı (çıkış istendi), aksi halde True
        """
        fields = [
            ("Ekran çapı", lambda: f'{settings.diagonal_in:.1f}"'),
            ("En/boy oranı", lambda: ASPECT_PRESETS[nearest_aspect_index(settings.aspect)][0]),
            ("Kamera, ekranın üst kenarının ne kadar üstünde", lambda: f"{settings.cam_gap_cm:.1f} cm"),
            ("Göz bebekleri arası mesafe", lambda: f"{settings.ipd_cm:.1f} cm"),
        ]

        def change(i, direction, big):
            if i == 0:
                settings.diagonal_in += direction * (1.0 if big else 0.1)
            elif i == 1:
                idx = (nearest_aspect_index(settings.aspect) + direction) % len(ASPECT_PRESETS)
                settings.aspect = ASPECT_PRESETS[idx][1]
            elif i == 2:
                settings.cam_gap_cm += direction * (1.0 if big else 0.5)
            elif i == 3:
                settings.ipd_cm += direction * (0.5 if big else 0.1)
            for name in ("diagonal_in", "cam_gap_cm", "ipd_cm"):   # kayan nokta birikmesini engelle
                setattr(settings, name, round(getattr(settings, name), 1))
            settings.clamp()

        sel = 0
        clock = pygame.time.Clock()
        while True:
            for e in pygame.event.get():
                if e.type == QUIT:
                    return False
                if e.type == VIDEORESIZE:
                    self.renderer.resize((e.w, e.h))
                elif e.type == KEYDOWN:
                    if e.key in (K_ESCAPE, K_RETURN, K_KP_ENTER):
                        return True
                    elif e.key == K_UP:
                        sel = (sel - 1) % len(fields)
                    elif e.key == K_DOWN:
                        sel = (sel + 1) % len(fields)
                    elif e.key in (K_LEFT, K_RIGHT):
                        change(sel, 1 if e.key == K_RIGHT else -1, bool(e.mod & KMOD_SHIFT))
            w, h = self.renderer.size
            glClearColor(0.02, 0.02, 0.06, 1.0)
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
            glEnable(GL_BLEND)
            glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
            _blit_text(self.title_font, "Ayarlar", TITLE_COLOR, 80, 50, h)
            for i, (name, value) in enumerate(fields):
                y = 150 + i * 62
                color = SELECTED_COLOR if i == sel else TEXT_COLOR
                tw, _ = _blit_text(self.item_font, f"{'>' if i == sel else ' '}  {name}:", color, 80, y, h)
                _blit_text(self.item_font, value(), color, 80 + tw + 24, y, h)
            sw, sh = settings.screen_cm()
            _blit_text(self.small_font, f"Hesaplanan ekran boyutu: {sw:.1f} x {sh:.1f} cm   "
                                        f"(kamera ekran merkezinin {settings.cam_offset_y_cm():.1f} cm üstünde)",
                       DIM_COLOR, 80, 150 + len(fields) * 62 + 20, h)
            _blit_text(self.small_font,
                       "Yukarı/Aşağı: alan seç    Sol/Sağ: değiştir (Shift: büyük adım)    Enter / ESC: kaydet ve dön",
                       DIM_COLOR, 80, h - 60, h)
            pygame.display.flip()
            clock.tick(30)

    def show_loading(self, item):
        """Model yüklenirken (birkaç saniye sürebilir) bilgi ekranı."""
        w, h = self.renderer.size
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        _blit_text(self.title_font, f"{_label(item)[0]} yükleniyor...", TITLE_COLOR, 80, h // 2 - 40, h)
        pygame.display.flip()
        pygame.event.pump()
