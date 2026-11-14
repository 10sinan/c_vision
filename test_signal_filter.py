"""signal_filter.py için testler. Çalıştır: python -m unittest test_signal_filter -v"""

import random
import statistics
import unittest

from signal_filter import OneEuroFilter


class OneEuroFilterTest(unittest.TestCase):
    def test_first_sample_passes_through(self):
        f = OneEuroFilter((1.0,), (0.0,))
        self.assertEqual(f.update((5.0,), 0.0), (5.0,))

    def test_reduces_noise_on_still_signal(self):
        rng = random.Random(0)
        f = OneEuroFilter((0.3,), (0.05,))
        raw, out = [], []
        for i in range(300):
            x = 60.0 + rng.gauss(0.0, 1.2)       # kamera Z gürültüsüne benzer
            raw.append(x)
            out.append(f.update((x,), i / 30.0)[0])
        self.assertLess(statistics.pstdev(out[60:]), statistics.pstdev(raw[60:]) * 0.3)

    def test_follows_movement(self):
        f = OneEuroFilter((1.0,), (3.0,))
        for i in range(60):                       # 2 saniyede 0 -> 1 kayma, sonra sabit
            t = i / 30.0
            f.update((min(1.0, t),), t)
        for i in range(60, 90):
            f.update((1.0,), i / 30.0)
        self.assertAlmostEqual(f.state[0], 1.0, places=2)

    def test_repeated_timestamp_is_ignored(self):
        f = OneEuroFilter((1.0,), (0.0,))
        f.update((0.0,), 1.0)
        self.assertEqual(f.update((10.0,), 1.0), (0.0,))

    def test_predict_extrapolates_moving_signal(self):
        f = OneEuroFilter((1.0, 1.0), (3.0, 3.0))
        for i in range(60):                       # x sabit hızla artar, y sabit
            f.update((i / 30.0, 0.0), i / 30.0)
        t_last = 59 / 30.0
        x_pred, y_pred = f.predict(t_last + 0.02, horizon=0.05)
        self.assertGreater(x_pred, f.state[0])    # hareket yönünde ileri
        self.assertAlmostEqual(y_pred, 0.0, places=6)

    def test_predict_respects_horizon_and_axes(self):
        f = OneEuroFilter((1.0, 1.0), (3.0, 3.0))
        for i in range(60):
            f.update((i / 30.0, i / 30.0), i / 30.0)
        t_last = 59 / 30.0
        far = f.predict(t_last + 10.0, horizon=0.05)
        near = f.predict(t_last + 0.05, horizon=0.05)
        self.assertAlmostEqual(far[0], near[0])   # ufuk sınırı
        only_x = f.predict(t_last + 0.05, horizon=0.05, axes=(0,))
        self.assertEqual(only_x[1], f.state[1])   # y tahmin edilmedi


if __name__ == "__main__":
    unittest.main()
