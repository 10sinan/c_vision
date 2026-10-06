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
