"""
Ekran üstü ayar paneli: klavye tuşlarının yaptığı işleri fareyle yapar.

Kapalıyken sağ üstte küçük bir "Ayarlar" düğmesi durur; tıklanınca (veya Tab) panel açılır.
Satır türleri:
  "step"    değer + [-] [+] düğmeleri (fare tekerleği de çalışır)
  "toggle"  aç/kapa (satırın tamamı tıklanır)
  "action"  tek seferlik eylem düğmesi

Yerleşim ve tıklama testi (rect / hit / click / wheel) OpenGL ve pencere gerektirmez; yalnızca
draw() GL kullanır. Panel bir pygame yüzeyine çizilir ve tek glDrawPixels ile basılır; yüzey
yalnızca içerik değişince yeniden üretilir.
"""

import pygame

PANEL_W = 380
MARGIN = 12          # pencere kenarından boşluk
HEADER_H = 44
ROW_H = 34           # pencere kısaysa küçülür (en az MIN_ROW_H)
MIN_ROW_H = 20
BTN_W = 34
PAD = 12
TAB_W, TAB_H = 150, 32

BG = (10, 12, 26, 225)
BTN = (40, 46, 72, 255)
BTN_HOVER = (70, 82, 130, 255)
ON = (40, 120, 70, 255)
ON_HOVER = (55, 155, 90, 255)
TEXT = (215, 215, 225)
DIM = (105, 105, 125)
TITLE = (120, 220, 255)
VALUE = (255, 200, 60)


class Row:
    def __init__(self, label, kind, change, value=None, enabled=None):
        """
        :param kind: "step" | "toggle" | "action"
        :param change: change(direction) — step için -1/+1, diğerleri için +1
        :param value: step: metin döndürür; toggle: bool döndürür; action: gerekmez
        :param enabled: verilirse False döndürdüğünde satır soluk çizilir ve tıklanmaz
        """
        self.label, self.kind, self.change, self.value, self.enabled = label, kind, change, value, enabled

    def is_enabled(self):
        return self.enabled is None or bool(self.enabled())


class Panel:
    def __init__(self, rows):
        self.rows = list(rows)
        self.open = False
        self._fonts = None
        self._cache_key = None
        self._cache = None

    # ------------------------------------------------------------------ yerleşim (saf)
    def _row_h(self, win_size):
        avail = win_size[1] - 2 * MARGIN - HEADER_H - PAD
        return max(MIN_ROW_H, min(ROW_H, avail // max(1, len(self.rows))))

    def rect(self, win_size):
        """:return: (x, y, w, h) — açıksa panelin, kapalıysa "Ayarlar" düğmesinin alanı"""
        if not self.open:
            return win_size[0] - MARGIN - TAB_W, MARGIN, TAB_W, TAB_H
        h = HEADER_H + len(self.rows) * self._row_h(win_size) + PAD
        return win_size[0] - MARGIN - PANEL_W, MARGIN, PANEL_W, h

    def hit(self, pos, win_size):
        """
        :return: None (panelin dışı) | ("tab",) | ("close",) | ("panel",) boş alan |
                 ("row", i, yön): yön step satırında -1/+1 (düğmeler) ya da 0 (satırın geri kalanı),
                 toggle/action satırında her zaman +1
        """
        if pos is None:
            return None
        x, y, w, h = self.rect(win_size)
        px, py = pos[0] - x, pos[1] - y
        if not (0 <= px < w and 0 <= py < h):
            return None
        if not self.open:
            return ("tab",)
        if py < HEADER_H:
            return ("close",) if px >= w - PAD - BTN_W else ("panel",)
        i = (py - HEADER_H) // self._row_h(win_size)
        if i >= len(self.rows):
            return ("panel",)
        if self.rows[i].kind != "step":
            return ("row", i, 1)
        if px >= w - PAD - BTN_W:
            return ("row", i, 1)
        if w - PAD - 2 * BTN_W - 4 <= px < w - PAD - BTN_W - 4:
            return ("row", i, -1)
        return ("row", i, 0)

    def click(self, pos, win_size):
        """:return: True ise tıklama panele aitti (sahneye iletilmemeli)."""
        h = self.hit(pos, win_size)
        if h is None:
            return False
        if h[0] == "tab":
            self.open = True
        elif h[0] == "close":
            self.open = False
        elif h[0] == "row" and h[2] != 0 and self.rows[h[1]].is_enabled():
            self.rows[h[1]].change(h[2])
        return True

    def wheel(self, pos, win_size, dy):
        """Fare tekerleği: üzerinde durulan step satırını değiştirir. :return: True ise panele aitti."""
        h = self.hit(pos, win_size)
        if h is None:
            return False
        if h[0] == "row" and dy and self.rows[h[1]].kind == "step" and self.rows[h[1]].is_enabled():
            self.rows[h[1]].change(1 if dy > 0 else -1)
        return True

    # ------------------------------------------------------------------ çizim
    def _render(self, win_size, hover):
        if self._fonts is None:
            self._fonts = pygame.font.Font(None, 30), pygame.font.Font(None, 24)
        title_font, font = self._fonts
        _, _, w, h = self.rect(win_size)
        surf = pygame.Surface((w, h), pygame.SRCALPHA)

        def text(s, color, x, cy, f=font, align="left"):
            t = f.render(s, True, color)
            tx = x if align == "left" else (x - t.get_width() if align == "right" else x - t.get_width() // 2)
            surf.blit(t, (tx, cy - t.get_height() // 2))

        def box(rect, color):
            pygame.draw.rect(surf, color, rect, border_radius=6)

        if not self.open:
            box((0, 0, w, h), BTN_HOVER if hover else BG)
            text("Ayarlar (Tab)", TEXT, w // 2, h // 2, align="center")
            return surf

        box((0, 0, w, h), BG)
        text("Ayarlar", TITLE, PAD, HEADER_H // 2, title_font)
        box((w - PAD - BTN_W, 7, BTN_W, HEADER_H - 14), BTN_HOVER if hover == ("close",) else BTN)
        text("x", TEXT, w - PAD - BTN_W // 2, HEADER_H // 2, align="center")

        rh = self._row_h(win_size)
        for i, row in enumerate(self.rows):
            top = HEADER_H + i * rh
            cy = top + rh // 2
            on = row.is_enabled()
            hov = hover[2] if (on and hover and hover[0] == "row" and hover[1] == i) else None
            label_color = TEXT if on else DIM
            if row.kind == "action":
                box((PAD, top + 2, w - 2 * PAD, rh - 4), BTN_HOVER if hov else BTN)
                text(row.label, label_color, w // 2, cy, align="center")
                continue
            text(row.label, label_color, PAD, cy)
            if row.kind == "toggle":
                state = bool(row.value())
                color = (ON_HOVER if hov else ON) if state else (BTN_HOVER if hov else BTN)
                bw = 2 * BTN_W + 4
                box((w - PAD - bw, top + 3, bw, rh - 6), color if on else BTN)
                text("Açık" if state else "Kapalı", label_color, w - PAD - bw // 2, cy, align="center")
            else:
                for d, sym, bx in ((-1, "-", w - PAD - 2 * BTN_W - 4), (1, "+", w - PAD - BTN_W)):
                    box((bx, top + 3, BTN_W, rh - 6), BTN_HOVER if hov == d else BTN)
                    text(sym, label_color, bx + BTN_W // 2, cy, align="center")
                text(str(row.value()), VALUE if on else DIM, w - PAD - 2 * BTN_W - 16, cy, align="right")
        return surf

    def draw(self, win_size, mouse_pos=None):
        """Paneli (ya da kapalıyken düğmesini) çizilmiş sahnenin üstüne basar; flip'ten önce çağrılır."""
        from OpenGL.GL import (glWindowPos2d, glDrawPixels, glEnable, glBlendFunc, GL_BLEND, GL_SRC_ALPHA,
                               GL_ONE_MINUS_SRC_ALPHA, GL_RGBA, GL_UNSIGNED_BYTE)
        hover = self.hit(mouse_pos, win_size)
        values = tuple((r.value() if r.value else None, r.is_enabled()) for r in self.rows) if self.open else ()
        key = (self.open, tuple(win_size), hover, values)
        if key != self._cache_key:
            surf = self._render(win_size, hover)
            self._cache = pygame.image.tobytes(surf, "RGBA", True)   # OpenGL alttan başlar -> dikey çevrik
            self._cache_key = key
        x, y, w, h = self.rect(win_size)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glWindowPos2d(x, win_size[1] - y - h)
        glDrawPixels(w, h, GL_RGBA, GL_UNSIGNED_BYTE, self._cache)
