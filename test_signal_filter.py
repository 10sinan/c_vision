"""signal_filter.py için testler. Çalıştır: python -m unittest test_signal_filter -v"""

import random
import statistics
import unittest

from signal_filter import OneEuroFilter, OutlierGate, decay_toward, blend


class OutlierGateTest(unittest.TestCase):
    def test_normal_motion_is_accepted(self):
        g = OutlierGate()
        for i in range(30):                       # kare başına 0.02 kayma, Z yavaş değişir
            self.assertTrue(g.accept((i * 0.02, 0.0, 60.0 + i * 0.3), i / 30.0))

    def test_single_spike_is_rejected(self):
        g = OutlierGate()
        self.assertTrue(g.accept((0.0, 0.0, 60.0), 0.0))
        self.assertFalse(g.accept((0.6, 0.0, 60.0), 1 / 30.0))    # X sıçraması
        self.assertFalse(g.accept((0.0, 0.0, 90.0), 2 / 30.0))    # Z sıçraması
        self.assertTrue(g.accept((0.01, 0.0, 61.0), 3 / 30.0))    # normale döndü

    def test_persistent_jump_is_accepted(self):
        g = OutlierGate(max_rejects=3)
        g.accept((0.0, 0.0, 60.0), 0.0)
        results = [g.accept((0.6, 0.0, 60.0), (i + 1) / 30.0) for i in range(4)]
        self.assertEqual(results, [False, False, False, True])
        self.assertTrue(g.accept((0.61, 0.0, 60.0), 5 / 30.0))    # yeni konum artık referans

    def test_long_gap_skips_comparison(self):
        g = OutlierGate(max_gap_s=0.2)
        g.accept((0.0, 0.0, 60.0), 0.0)
        self.assertTrue(g.accept((0.6, 0.0, 60.0), 1.0))

    def test_reset_forgets_reference(self):
        g = OutlierGate()
        g.accept((0.0, 0.0, 60.0), 0.0)
        g.reset()
        self.assertTrue(g.accept((0.9, 0.0, 60.0), 1 / 30.0))


class HelpersTest(unittest.TestCase):
    def test_decay_is_frame_rate_independent(self):
        target = (0.0, 0.0, 60.0)
        a = b = (1.0, -1.0, 100.0)
        for _ in range(60):
            a = decay_toward(a, target, 1 / 60.0, 0.5)
        for _ in range(30):
            b = decay_toward(b, target, 1 / 30.0, 0.5)
        for x, y in zip(a, b):
            self.assertAlmostEqual(x, y, places=9)
        self.assertAlmostEqual(a[0], 2.718281828 ** -2, places=6)   # 1 sn = 2 tau

    def test_blend_endpoints_and_clamp(self):
        a, b = (0.0, 10.0), (1.0, 20.0)
        self.assertEqual(blend(a, b, -1.0), a)
        self.assertEqual(blend(a, b, 0.0), a)
        self.assertEqual(blend(a, b, 0.5), (0.5, 15.0))
        self.assertEqual(blend(a, b, 2.0), b)


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

    def test_predict_fades_when_measurements_stop(self):
        f = OneEuroFilter((1.0,), (3.0,))
        for i in range(60):
            f.update((i / 30.0,), i / 30.0)
        t_last = 59 / 30.0
        fresh = f.predict(t_last + 0.05, horizon=0.05, fade=0.15)[0]
        self.assertAlmostEqual(fresh, f.predict(t_last + 0.05, horizon=0.05)[0])
        stale = f.predict(t_last + 0.2, horizon=0.05, fade=0.15)[0]
        old = f.predict(t_last + 2.0, horizon=0.05, fade=0.15)[0]
        self.assertLess(stale, fresh)             # tahmin geri çekiliyor
        self.assertGreater(stale, f.state[0])
        self.assertAlmostEqual(old, f.state[0], places=4)


if __name__ == "__main__":
    unittest.main()
