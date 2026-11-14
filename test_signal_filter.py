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


if __name__ == "__main__":
    unittest.main()
