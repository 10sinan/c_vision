"""calibration.py'nin saf matematiği için testler (kamera/OpenGL gerektirmez).
Çalıştır: python -m unittest test_calibration -v"""

import random
import unittest

from calibration import focal_mult_from_measurement, summarize, evaluate, MIN_SAMPLES

FRAME_W = 640
IPD_CM = 6.3


def ipd_px_at(z_cm, focal_px):
    """Pinhole modeli: ipd_px = f * ipd_cm / z"""
    return focal_px * IPD_CM / z_cm


class FocalMultTest(unittest.TestCase):
    def test_inverts_depth_formula(self):
        true_focal_px = 700.0
        ipd_px = ipd_px_at(60.0, true_focal_px)
        fm = focal_mult_from_measurement(ipd_px, 60.0, IPD_CM, FRAME_W)
        self.assertAlmostEqual(fm * FRAME_W, true_focal_px, places=6)

    def test_default_focal_gives_back_default_mult(self):
        ipd_px = ipd_px_at(60.0, 0.85 * FRAME_W)
        self.assertAlmostEqual(focal_mult_from_measurement(ipd_px, 60.0, IPD_CM, FRAME_W), 0.85)

    def test_rejects_non_positive_inputs(self):
        with self.assertRaises(ValueError):
            focal_mult_from_measurement(0.0, 60.0, IPD_CM, FRAME_W)


class EvaluateTest(unittest.TestCase):
    def _noisy(self, z, focal_px, n=60, sigma=0.5, seed=1):
        rng = random.Random(seed)
        return [ipd_px_at(z, focal_px) + rng.gauss(0, sigma) for _ in range(n)]

    def test_good_measurement(self):
        fm, msg = evaluate(self._noisy(60, 600), 60, IPD_CM, FRAME_W)
        self.assertIsNotNone(fm)
        self.assertAlmostEqual(fm, 600 / FRAME_W, delta=0.01)

    def test_robust_to_outliers(self):
        samples = self._noisy(60, 600)
        samples[3], samples[10] = samples[3] * 3, samples[10] * 0.2   # iki aykırı değer
        fm, _ = evaluate(samples, 60, IPD_CM, FRAME_W)
        self.assertAlmostEqual(fm, 600 / FRAME_W, delta=0.01)

    def test_too_few_samples(self):
        fm, msg = evaluate(self._noisy(60, 600, n=MIN_SAMPLES - 1), 60, IPD_CM, FRAME_W)
        self.assertIsNone(fm)

    def test_shaky_measurement_rejected(self):
        samples = self._noisy(60, 600, sigma=8.0)    # kullanıcı hareket etti
        fm, msg = evaluate(samples, 60, IPD_CM, FRAME_W)
        self.assertIsNone(fm)
        self.assertIn("oynak", msg)

    def test_out_of_range_result_rejected(self):
        fm, _ = evaluate([5.0] * 60, 60, IPD_CM, FRAME_W)   # 5 px göz mesafesi: imkansız derecede küçük
        self.assertIsNone(fm)


class SummarizeTest(unittest.TestCase):
    def test_median_and_spread(self):
        med, spread = summarize([100, 100, 100, 100, 1000])
        self.assertEqual(med, 100)
        self.assertEqual(spread, 0.0)


if __name__ == "__main__":
    unittest.main()
