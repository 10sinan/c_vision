"""
Signal Filter Modülü - Normalizasyon ve EMA Filtreleme (Aşama 2)
----------------------------------------------------------------
HeadTracker'dan gelen ham piksel koordinatlarını [-1, 1] aralığına
normalize eder ve Exponential Moving Average (EMA) ile yumuşatır.
"""

import math


def _clamp(value, lo, hi):
    return max(lo, min(hi, value))


class PoseNormalizer:
    """
    Piksel (x, y) -> ekran merkezine göre [-1, 1] aralığı.
      x_n = (x_px - W/2) / (W/2)    -> sağ: +1, sol: -1
      y_n = (H/2 - y_px) / (H/2)    -> yukarı: +1, aşağı: -1 (OpenGL ile uyumlu)
    Z (cm) fiziksel anlamını korur; yalnızca [z_min, z_max] aralığına kırpılır.
    """

    def __init__(self, frame_width, frame_height, z_min_cm=20.0, z_max_cm=160.0):
        self.w = frame_width
        self.h = frame_height
        self.z_min = z_min_cm
        self.z_max = z_max_cm

    def normalize(self, raw_pose):
        """:return: (x_n, y_n, z_cm)"""
        x_n = (raw_pose['x_px'] - self.w / 2.0) / (self.w / 2.0)
        y_n = (self.h / 2.0 - raw_pose['y_px']) / (self.h / 2.0)
        z = _clamp(raw_pose['z_cm'], self.z_min, self.z_max)
        return _clamp(x_n, -1.0, 1.0), _clamp(y_n, -1.0, 1.0), z


class EMAFilter:
    """
    Exponential Moving Average:  S_t = alpha * X_t + (1 - alpha) * S_(t-1)

    alpha küçük -> daha pürüzsüz ama gecikmeli
    alpha büyük -> daha hızlı tepki ama titrek
    Vektör (x, y, z) için bileşen bazında uygulanır.
    """

    def __init__(self, alpha=0.8):
        if not 0.0 < alpha <= 1.0:
            raise ValueError("alpha (0, 1] aralığında olmalı.")
        self.alpha = alpha
        self.state = None

    def update(self, values):
        if self.state is None:
            self.state = tuple(values)
        else:
            a = self.alpha
            self.state = tuple(a * v + (1.0 - a) * s for v, s in zip(values, self.state))
        return self.state

    def reset(self):
        self.state = None


class OneEuroFilter:
    """
    One Euro filtresi (Casiez ve ark., 2012) - hıza duyarlı alçak geçiren filtre.

    Baş sabitken kesim frekansı düşük kalır -> titreme güçlü bastırılır.
    Baş hızlandıkça kesim frekansı yükselir -> gecikme azalır.
      fc = min_cutoff + beta * |filtrelenmiş hız|
      alpha = 1 / (1 + 1 / (2*pi*fc*dt))

    Vektör için bileşen bazında; min_cutoff ve beta bileşen başına ayrı verilebilir
    (ör. Z derinliği X/Y'den daha gürültülü olduğu için daha güçlü filtrelenir).
    update() yalnızca YENİ ölçümlerle ve ölçüm zaman damgasıyla çağrılmalıdır.
    """

    def __init__(self, min_cutoff, beta, d_cutoff=1.0):
        self.min_cutoff = list(min_cutoff)
        self.beta = list(beta)
        self.d_cutoff = d_cutoff
        self.state = None
        self._dx = None
        self._t = None

    @staticmethod
    def _alpha(cutoff, dt):
        return 1.0 / (1.0 + 1.0 / (2.0 * math.pi * cutoff * dt))

    def update(self, values, t):
        values = tuple(values)
        if self.state is None or self._t is None:
            self.state, self._dx, self._t = values, (0.0,) * len(values), t
            return self.state
        dt = t - self._t
        if dt <= 0:
            return self.state
        self._t = t
        a_d = self._alpha(self.d_cutoff, dt)
        new_state, new_dx = [], []
        for i, (x, s, dx_prev) in enumerate(zip(values, self.state, self._dx)):
            dx = a_d * ((x - s) / dt) + (1.0 - a_d) * dx_prev
            fc = self.min_cutoff[i] + self.beta[i] * abs(dx)
            a = self._alpha(fc, dt)
            new_state.append(a * x + (1.0 - a) * s)
            new_dx.append(dx)
        self.state, self._dx = tuple(new_state), tuple(new_dx)
        return self.state

    def set_state(self, values, t=None):
        """Filtre çıktısını doğrudan ayarlar (ör. yüz kaybolunca merkeze dönüş)."""
        self.state = tuple(values)
        self._dx = (0.0,) * len(self.state)
        if t is not None:
            self._t = t

    def reset(self):
        self.state = self._dx = self._t = None
