"""
Prosedürel efektler (OpenGL'siz): ekranın ÖNÜNDE süzülen yapraklar ve kelebekler + sahne boyunca
sürüklenen toz/polen parçacıkları.

Modellerde animasyon yok; hareketi buradaki efektler verir. Tüm konumlar zamanın saf fonksiyonudur
(durum tutulmaz, yalnızca t ilerler) -> sınırlar garanti, test edilebilir, dt'den bağımsız.

Koordinatlar renderer ile aynı: cm, ekran düzlemi z=0 merkezde, kullanıcı z>0 tarafında.
"""

import math

import numpy as np

REF_EYE_Z = 60.0            # kullanıcının tipik göz mesafesi (görünür alan hesabı için)
FG_Z_RANGE = (4.0, 26.0)    # ekranın önünde kalan nesnelerin z aralığı
DUST_Z_RANGE = (-70.0, 28.0)
VIEW_MARGIN = 1.15          # görünür alanın biraz dışına taşmalarına izin ver

BASE_DUST, BASE_LEAVES, BASE_BUTTERFLIES = 160, 10, 5

LEAF_COLORS = np.array([(0.35, 0.65, 0.20), (0.55, 0.75, 0.25),
                        (0.85, 0.65, 0.15), (0.80, 0.40, 0.10)])
BUTTERFLY_COLORS = np.array([(0.95, 0.55, 0.10), (0.20, 0.50, 0.95), (0.95, 0.90, 0.30)])
LIGHT_DIR = np.array([0.3, 0.6, 1.0]) / np.linalg.norm([0.3, 0.6, 1.0])


def _wrap(x, half):
    """x'i [-half, half) aralığına sarar (dizi destekli)."""
    return np.mod(x + half, 2.0 * half) - half


def _rotation(rx, ry, rz):
    """Euler açılarından (radyan) 3x3 döndürme matrisi (Rz @ Ry @ Rx)."""
    cx, sx, cy, sy, cz, sz = math.cos(rx), math.sin(rx), math.cos(ry), math.sin(ry), math.cos(rz), math.sin(rz)
    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def visible_half_extent(screen_half, z, eye_z=REF_EYE_Z):
    """
    Göz (0,0,eye_z) iken, z düzlemindeki bir noktanın ekran penceresinde kalması için |x|'in
    alabileceği en büyük değer (benzer üçgenler: ekran kenarı z düzlemine ölçeklenir).
    """
    return screen_half * (eye_z - z) / eye_z * VIEW_MARGIN


class Effects:
    def __init__(self, screen_w, screen_h, density=1.0, seed=7):
        self.seed = seed
        self.t = 0.0
        self.screen_w, self.screen_h = screen_w, screen_h
        self.density = density
        self._build()

    # ------------------------------------------------------------------ kurulum
    def set_screen(self, screen_w, screen_h):
        self.screen_w, self.screen_h = screen_w, screen_h

    def set_density(self, density):
        density = min(max(float(density), 0.0), 3.0)
        if density != self.density:
            self.density = density
            self._build()

    def _build(self):
        rng = np.random.default_rng(self.seed)
        n_dust = int(BASE_DUST * self.density)
        n_leaf = int(round(BASE_LEAVES * self.density))
        n_bfly = int(round(BASE_BUTTERFLIES * self.density))

        # Toz: [0,1) üç birim kutuda rastgele; gerçek konuma çizim anında ölçeklenir
        self._dust_u = rng.random((n_dust, 3))
        self._dust_vel = rng.uniform(-0.6, 0.6, (n_dust, 3)) + np.array([0.0, 0.5, 0.0])   # hafif yukarı sürüklenme
        self._dust_phase = rng.uniform(0, 2 * math.pi, (n_dust, 3))

        # Yapraklar
        self._leaf_u = rng.random((n_leaf, 3))                      # x, y başlangıcı, z
        self._leaf_size = rng.uniform(1.2, 2.2, n_leaf)
        self._leaf_fall = rng.uniform(1.0, 2.2, n_leaf)             # cm/sn düşüş
        self._leaf_sway = rng.uniform(1.0, 3.0, (n_leaf, 2))        # sallanma genliği (x, z)
        self._leaf_sway_f = rng.uniform(0.3, 0.8, n_leaf)
        self._leaf_rot0 = rng.uniform(0, 2 * math.pi, (n_leaf, 3))
        self._leaf_rot_v = rng.uniform(-1.2, 1.2, (n_leaf, 3))      # rad/sn
        self._leaf_color = rng.integers(0, len(LEAF_COLORS), n_leaf)

        # Kelebekler
        self._bf_u = rng.random((n_bfly, 3))
        self._bf_amp = rng.uniform(1.5, 4.0, (n_bfly, 3))
        self._bf_freq = rng.uniform(0.25, 0.7, (n_bfly, 3))
        self._bf_phase = rng.uniform(0, 2 * math.pi, (n_bfly, 3))
        self._bf_drift = rng.uniform(0.8, 2.0, n_bfly) * rng.choice([-1, 1], n_bfly)   # cm/sn yatay
        self._bf_flap_f = rng.uniform(2.0, 3.5, n_bfly)             # Hz (görsel olarak okunur çırpma)
        self._bf_size = rng.uniform(1.4, 2.2, n_bfly)
        self._bf_color = rng.integers(0, len(BUTTERFLY_COLORS), n_bfly)

    @property
    def counts(self):
        return len(self._dust_u), len(self._leaf_u), len(self._bf_u)

    def update(self, dt):
        self.t += max(0.0, dt)

    # ------------------------------------------------------------------ konumlar (t'nin saf fonksiyonu)
    def _z_of(self, u, z_range):
        return z_range[0] + u * (z_range[1] - z_range[0])

    def dust_positions(self):
        """:return: (N, 3) toz parçacığı konumları"""
        t = self.t
        hw, hh = self.screen_w / 2.0, self.screen_h / 2.0
        z = self._z_of(self._dust_u[:, 2], DUST_Z_RANGE)
        # görünür alan derinliğe göre değişir: uzaktaki parçacıklar daha geniş alana yayılır
        xh = np.maximum(hw * (REF_EYE_Z - z) / REF_EYE_Z * VIEW_MARGIN, hw * 0.3)
        yh = np.maximum(hh * (REF_EYE_Z - z) / REF_EYE_Z * VIEW_MARGIN, hh * 0.3)
        sway = 0.4 * np.sin(0.5 * t + self._dust_phase)
        x = _wrap((self._dust_u[:, 0] * 2 - 1) * xh + self._dust_vel[:, 0] * t + sway[:, 0], xh)
        y = _wrap((self._dust_u[:, 1] * 2 - 1) * yh + self._dust_vel[:, 1] * t + sway[:, 1], yh)
        z = z + sway[:, 2]
        return np.stack([x, y, z], axis=1)

    def leaf_positions(self):
        """:return: (M, 3) yaprak konumları (aşağı düşer, üstten yeniden başlar)"""
        t = self.t
        hw, hh = self.screen_w / 2.0, self.screen_h / 2.0
        z0 = self._z_of(self._leaf_u[:, 2], FG_Z_RANGE)
        xh = visible_half_extent(hw, z0)
        yh = visible_half_extent(hh, z0)
        y = _wrap((self._leaf_u[:, 1] * 2 - 1) * yh - self._leaf_fall * t, yh)
        sway = self._leaf_sway * np.sin(self._leaf_sway_f[:, None] * t + self._leaf_rot0[:, :2])
        x = (self._leaf_u[:, 0] * 2 - 1) * xh * 0.8 + sway[:, 0]
        z = z0 + 0.3 * sway[:, 1]
        return np.stack([x, y, z], axis=1)

    def butterfly_positions(self):
        """:return: (B, 3) kelebek konumları (Lissajous yolu + yavaş yatay sürüklenme)"""
        t = self.t
        hw, hh = self.screen_w / 2.0, self.screen_h / 2.0
        z0 = self._z_of(self._bf_u[:, 2], FG_Z_RANGE)
        xh = visible_half_extent(hw, z0)
        yh = visible_half_extent(hh, z0)
        liss = self._bf_amp * np.sin(self._bf_freq * 2 * math.pi * t * 0.5 + self._bf_phase)
        ax = _wrap((self._bf_u[:, 0] * 2 - 1) * xh + self._bf_drift * t, xh)
        ay = (self._bf_u[:, 1] * 2 - 1) * yh * 0.7
        return np.stack([ax + liss[:, 0], ay + liss[:, 1], z0 + 0.5 * liss[:, 2]], axis=1)

    # ------------------------------------------------------------------ şekiller
    def leaf_quad(self, i):
        """
        :return: (köşeler (4,3) dünya koordinatı, renk (3,)) — yaprak elmas biçimi, ışığa göre gölgelenmiş.
        """
        s = self._leaf_size[i]
        base = np.array([(0, s, 0), (-0.45 * s, 0, 0), (0, -s, 0), (0.45 * s, 0, 0)], dtype=float)
        ang = self._leaf_rot0[i] + self._leaf_rot_v[i] * self.t
        R = _rotation(*ang)
        verts = base @ R.T + self.leaf_positions()[i]
        normal = R @ np.array([0.0, 0.0, 1.0])
        shade = 0.45 + 0.55 * abs(float(normal @ LIGHT_DIR))
        return verts, LEAF_COLORS[self._leaf_color[i]] * shade

    def butterfly_wings(self, i):
        """
        :return: (üçgenler (4,3,3) dünya koordinatı [sol üst, sol alt, sağ üst, sağ alt kanat], renk (3,)).
        Kanatlar gövde ekseni (dikey) etrafında çırpar; kelebek kullanıcıya doğru hafif eğik durur.
        """
        s = self._bf_size[i]
        flap = 1.0 * math.sin(2 * math.pi * self._bf_flap_f[i] * self.t + self._bf_phase[i, 0])  # [-1, 1]
        angle = 0.15 + 1.05 * (0.5 + 0.5 * flap)          # 0.15..1.2 rad açıklık
        tris = []
        tilt = _rotation(-0.9, 0, 0)                      # kullanıcıya doğru eğim
        for side in (-1, 1):
            ry = _rotation(0, -side * angle, 0)           # kanat gövde ekseni etrafında kalkar
            upper = np.array([(0, 0.1 * s, 0), (side * s, 0.85 * s, 0), (side * 0.95 * s, 0.0, 0)], dtype=float)
            lower = np.array([(0, 0.0, 0), (side * 0.95 * s, 0.0, 0), (side * 0.6 * s, -0.75 * s, 0)], dtype=float)
            tris.append(upper @ ry.T @ tilt.T)
            tris.append(lower @ ry.T @ tilt.T)
        pos = self.butterfly_positions()[i]
        return np.array(tris) + pos, BUTTERFLY_COLORS[self._bf_color[i]]

    def butterfly_body(self, i):
        """:return: (2, 3) gövde çizgisinin uçları (dünya koordinatı)"""
        s = self._bf_size[i]
        tilt = _rotation(-0.9, 0, 0)
        body = np.array([(0, 0.55 * s, 0), (0, -0.65 * s, 0)], dtype=float) @ tilt.T
        return body + self.butterfly_positions()[i]
