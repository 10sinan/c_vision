"""
Prosedürel efektler (OpenGL'siz). Tür (kind) seçilebilir:
  yaprak      ekranın ÖNÜNDE süzülen yapraklar + sahne boyunca sürüklenen toz/polen
  yağmur      eğik düşen damla çizgileri
  kar         yavaşça salınarak düşen taneler
  ateşböceği  yanıp sönen, ağır ağır gezinen ışık noktaları
Yağmur, kar ve ateşböcekleri sahnenin içinden ekranın önüne kadar tüm derinliğe yayılır.

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

BASE_DUST, BASE_LEAVES = 160, 10
BASE_RAIN, BASE_SNOW, BASE_FIREFLIES = 350, 220, 26
FIREFLY_Z_RANGE = (-45.0, 22.0)

# Efekt türleri (Effects.kind bu listenin indeksidir; <model>.view.json'da fx_kind olarak saklanır)
LEAVES, RAIN, SNOW, FIREFLIES = range(4)
KIND_NAMES = ("Yaprak", "Yağmur", "Kar", "Ateşböceği")

LEAF_COLORS = np.array([(0.35, 0.65, 0.20), (0.55, 0.75, 0.25),
                        (0.85, 0.65, 0.15), (0.80, 0.40, 0.10)])
LIGHT_DIR = np.array([0.4, 0.9, 0.5]) / np.linalg.norm([0.4, 0.9, 0.5])   # renderer.KEY_LIGHT_DIR ile aynı


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
    def __init__(self, screen_w, screen_h, density=1.0, seed=7, kind=LEAVES):
        self.seed = seed
        self.kind = kind
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

        # Diğer türler en sonda üretilir: toz/yaprak rastgele dizileri bunlardan etkilenmez
        n_rain = int(BASE_RAIN * self.density)
        self._rain_u = rng.random((n_rain, 3))
        self._rain_speed = rng.uniform(55.0, 85.0, n_rain)          # cm/sn (küçük pencerede okunur hız)
        self._rain_len = rng.uniform(1.8, 3.2, n_rain)              # çizgi boyu, cm

        n_snow = int(BASE_SNOW * self.density)
        self._snow_u = rng.random((n_snow, 3))
        self._snow_fall = rng.uniform(4.0, 9.0, n_snow)
        self._snow_amp = rng.uniform(0.8, 2.5, (n_snow, 2))         # salınım genliği (x, z)
        self._snow_freq = rng.uniform(0.3, 0.9, n_snow)
        self._snow_phase = rng.uniform(0, 2 * math.pi, (n_snow, 2))

        n_fly = int(round(BASE_FIREFLIES * self.density))
        self._fly_u = rng.random((n_fly, 3))
        self._fly_amp = rng.uniform(2.5, 7.0, (n_fly, 3))
        self._fly_freq = rng.uniform(0.04, 0.16, (n_fly, 3))        # Hz; ağır gezinme
        self._fly_phase = rng.uniform(0, 2 * math.pi, (n_fly, 3))
        self._fly_blink_f = rng.uniform(0.2, 0.5, n_fly)
        self._fly_blink_p = rng.uniform(0, 2 * math.pi, n_fly)

    def set_kind(self, kind):
        self.kind = int(kind) % len(KIND_NAMES)

    @property
    def counts(self):
        """:return: (toz, yaprak) sayısı; yaprak türü seçili değilse (0, 0)"""
        if self.kind != LEAVES:
            return 0, 0
        return len(self._dust_u), len(self._leaf_u)

    def _volume(self, u, z_range=DUST_Z_RANGE):
        """Birim küp koordinatları için (z, x yarı genişliği, y yarı yüksekliği): görünür alan derinlikle genişler."""
        hw, hh = self.screen_w / 2.0, self.screen_h / 2.0
        z = self._z_of(u[:, 2], z_range)
        xh = np.maximum(hw * (REF_EYE_Z - z) / REF_EYE_Z * VIEW_MARGIN, hw * 0.3)
        yh = np.maximum(hh * (REF_EYE_Z - z) / REF_EYE_Z * VIEW_MARGIN, hh * 0.3)
        return z, xh, yh

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

    RAIN_WIND = 0.22   # yatay hız / düşüş hızı (yağmurun eğimi)

    def rain_segments(self):
        """:return: (N, 2, 3) damla çizgileri [alt uç, üst uç]; yağmur seçili değilse boş"""
        if self.kind != RAIN or not len(self._rain_u):
            return np.zeros((0, 2, 3))
        t = self.t
        z, xh, yh = self._volume(self._rain_u)
        v = self._rain_speed
        x = _wrap((self._rain_u[:, 0] * 2 - 1) * xh + self.RAIN_WIND * v * t, xh)
        y = _wrap((self._rain_u[:, 1] * 2 - 1) * yh - v * t, yh)
        head = np.stack([x, y, z], axis=1)
        direction = np.array([self.RAIN_WIND, -1.0, 0.0]) / math.hypot(self.RAIN_WIND, 1.0)
        tail = head - direction * self._rain_len[:, None]
        return np.stack([head, tail], axis=1)

    def snow_positions(self):
        """:return: (N, 3) kar tanesi konumları; kar seçili değilse boş"""
        if self.kind != SNOW or not len(self._snow_u):
            return np.zeros((0, 3))
        t = self.t
        z, xh, yh = self._volume(self._snow_u)
        sway = self._snow_amp * np.sin(self._snow_freq[:, None] * 2 * math.pi * t + self._snow_phase)
        x = _wrap((self._snow_u[:, 0] * 2 - 1) * xh + sway[:, 0], xh)
        y = _wrap((self._snow_u[:, 1] * 2 - 1) * yh - self._snow_fall * t, yh)
        return np.stack([x, y, z + sway[:, 1]], axis=1)

    def firefly_points(self):
        """:return: (konumlar (N, 3), parlaklık (N,) 0..1); ateşböceği seçili değilse boş"""
        if self.kind != FIREFLIES or not len(self._fly_u):
            return np.zeros((0, 3)), np.zeros(0)
        t = self.t
        z, xh, yh = self._volume(self._fly_u, FIREFLY_Z_RANGE)
        wander = self._fly_amp * np.sin(self._fly_freq * 2 * math.pi * t + self._fly_phase)
        x = (self._fly_u[:, 0] * 2 - 1) * xh * 0.8 + wander[:, 0]
        y = (self._fly_u[:, 1] * 2 - 1) * yh * 0.8 + wander[:, 1]
        pos = np.stack([x, y, z + wander[:, 2]], axis=1)
        pulse = np.sin(self._fly_blink_f * 2 * math.pi * t + self._fly_blink_p)
        return pos, np.clip(pulse, 0.0, 1.0) ** 2      # yarı zamanda sönük, arada yumuşak parlama

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
