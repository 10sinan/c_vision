"""
Kullanıcıya özel ayarlar (OpenGL'siz, test edilebilir). `settings.json` proje klasöründe saklanır.

Ekran fiziksel boyutu çapraz uzunluktan (inç) ve en-boy oranından hesaplanır.
Webcam konumu ekranın üst kenarına göre (cm) tutulur; dikey ofset ekran boyutuna göre türetilir,
böylece ekran boyutu değişince kamera konumu da tutarlı kalır.
"""

import json
import math
import os
from dataclasses import dataclass, asdict, fields

SETTINGS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "settings.json")
CM_PER_INCH = 2.54

# Ayarlar ekranında döngüyle seçilebilen en-boy oranları: (ad, w/h)
ASPECT_PRESETS = (("16:9", 16 / 9), ("16:10", 16 / 10), ("3:2", 3 / 2), ("4:3", 4 / 3), ("21:9", 21 / 9))


def nearest_aspect_index(aspect):
    """:return: aspect'e en yakın ön ayarın indeksi"""
    return min(range(len(ASPECT_PRESETS)), key=lambda i: abs(ASPECT_PRESETS[i][1] - aspect))


@dataclass
class Settings:
    diagonal_in: float = 15.6        # ekran çapı (inç)
    aspect: float = 16 / 9           # genişlik / yükseklik
    cam_gap_cm: float = 1.0          # webcam, ekranın üst kenarının bu kadar üstünde
    ipd_cm: float = 6.3              # göz bebekleri arası mesafe (derinlik tahmini için)
    focal_mult: float = 0.85         # kamera odak uzaklığı çarpanı (kalibrasyonla ayarlanır)
    parallax_gain: float = 2.5       # baş hareketi çarpanı: 1 = fiziksel olarak doğru, >1 = güçlendirilmiş etki

    def screen_cm(self):
        """:return: (genişlik, yükseklik) cm; çapraz ve en-boy oranından."""
        h_in = self.diagonal_in / math.sqrt(1.0 + self.aspect ** 2)
        return self.aspect * h_in * CM_PER_INCH, h_in * CM_PER_INCH

    def cam_offset_y_cm(self):
        """Webcam'in ekran merkezinden yukarı uzaklığı (cm)."""
        return self.screen_cm()[1] / 2.0 + self.cam_gap_cm

    def clamp(self):
        """Makul sınırlara çeker (bozuk/elle düzenlenmiş dosyalara karşı)."""
        self.diagonal_in = min(max(self.diagonal_in, 5.0), 100.0)
        self.aspect = min(max(self.aspect, 1.0), 4.0)
        self.cam_gap_cm = min(max(self.cam_gap_cm, 0.0), 15.0)
        self.ipd_cm = min(max(self.ipd_cm, 4.5), 8.0)
        self.focal_mult = min(max(self.focal_mult, 0.2), 3.0)
        self.parallax_gain = min(max(self.parallax_gain, 0.5), 6.0)
        return self

    @classmethod
    def load(cls, path=SETTINGS_PATH, default_aspect=16 / 9):
        """
        :return: (Settings, dosya_vardı_mı). Dosya yoksa/bozuksa varsayılanlar döner.
        default_aspect: dosya yokken kullanılacak en-boy oranı (ör. masaüstü çözünürlüğünden).
        """
        s = cls(aspect=default_aspect)
        try:
            with open(path) as fh:
                data = json.load(fh)
        except (OSError, ValueError):
            return s.clamp(), False
        if not isinstance(data, dict):
            return s.clamp(), False
        for f in fields(cls):
            if f.name in data:
                try:
                    setattr(s, f.name, float(data[f.name]))
                except (TypeError, ValueError):
                    pass
        return s.clamp(), True

    def save(self, path=SETTINGS_PATH):
        with open(path, "w") as fh:
            json.dump(asdict(self), fh, indent=2)
