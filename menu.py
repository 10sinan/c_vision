"""
Açılış menüsü - proje klasöründeki 3B modelleri listeler, kullanıcı birini seçer.
OpenGL penceresinde çalışır: metin pygame.font ile çizilip glDrawPixels ile basılır.

Kullanım: ↑/↓ veya fare ile seç, Enter / tık ile aç, 1-9 doğrudan seçim, ESC çıkış.
"""

import glob
import os

import pygame
from pygame.locals import (QUIT, KEYDOWN, MOUSEMOTION, MOUSEBUTTONDOWN, VIDEORESIZE,
                           K_ESCAPE, K_q, K_UP, K_DOWN, K_RETURN, K_KP_ENTER, K_SPACE, K_f)
from OpenGL.GL import (glClear, glClearColor, glWindowPos2d, glDrawPixels, glEnable,
                       glBlendFunc, GL_COLOR_BUFFER_BIT, GL_DEPTH_BUFFER_BIT, GL_RGBA,
                       GL_UNSIGNED_BYTE, GL_BLEND, GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

MODEL_PATTERNS = ("*.glb", "*.gltf", "*.obj")
NO_MODEL = None  # menüde "model yok" (tel kafes oda) seçeneği

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
    size_mb = os.path.getsize(path) / (1024 * 1024)
    return pretty_name(path), f"{os.path.basename(path)}  ·  {size_mb:.0f} MB"


def _blit_text(font, text, color, x, y_top, win_h):
    """Metni pencerenin (x, y_top) noktasına (sol üst köşe referanslı) çizer."""
    surf = font.render(text, True, color)
    data = pygame.image.tobytes(surf, "RGBA", True)   # OpenGL alttan başlar -> dikey çevrik
    glWindowPos2d(x, win_h - y_top - surf.get_height())
    glDrawPixels(surf.get_width(), surf.get_height(), GL_RGBA, GL_UNSIGNED_BYTE, data)
    return surf.get_width(), surf.get_height()


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

    def show_loading(self, item):
        """Model yüklenirken (birkaç saniye sürebilir) bilgi ekranı."""
        w, h = self.renderer.size
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        _blit_text(self.title_font, f"{_label(item)[0]} yükleniyor...", TITLE_COLOR, 80, h // 2 - 40, h)
        pygame.display.flip()
        pygame.event.pump()
