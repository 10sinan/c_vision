"""settings.py için testler. Çalıştır: python -m unittest test_settings -v"""

import json
import os
import tempfile
import unittest

from settings import Settings, nearest_aspect_index, ASPECT_PRESETS


class ScreenSizeTest(unittest.TestCase):
    def test_default_matches_old_hardcoded_laptop(self):
        w, h = Settings().screen_cm()
        self.assertAlmostEqual(w, 34.5, delta=0.1)
        self.assertAlmostEqual(h, 19.4, delta=0.1)

    def test_diagonal_is_preserved(self):
        s = Settings(diagonal_in=27.0, aspect=16 / 10)
        w, h = s.screen_cm()
        self.assertAlmostEqual((w ** 2 + h ** 2) ** 0.5 / 2.54, 27.0, places=6)
        self.assertAlmostEqual(w / h, 1.6, places=6)

    def test_camera_offset_follows_screen_height(self):
        s = Settings()
        self.assertAlmostEqual(s.cam_offset_y_cm(), s.screen_cm()[1] / 2 + 1.0)
        self.assertAlmostEqual(s.cam_offset_y_cm(), 10.7, delta=0.1)   # eski CAMERA_OFFSET_Y_CM


class PersistenceTest(unittest.TestCase):
    def test_roundtrip(self):
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "s.json")
            Settings(diagonal_in=24.0, focal_mult=1.1).save(p)
            s, existed = Settings.load(p)
            self.assertTrue(existed)
            self.assertEqual((s.diagonal_in, s.focal_mult), (24.0, 1.1))

    def test_missing_file_uses_defaults_and_given_aspect(self):
        s, existed = Settings.load("/nonexistent/x.json", default_aspect=16 / 10)
        self.assertFalse(existed)
        self.assertAlmostEqual(s.aspect, 1.6)

    def test_corrupt_file_falls_back(self):
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "s.json")
            open(p, "w").write("{not json")
            s, existed = Settings.load(p)
            self.assertFalse(existed)
            self.assertEqual(s.diagonal_in, 15.6)

    def test_bad_values_are_ignored_or_clamped(self):
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "s.json")
            json.dump({"diagonal_in": "abc", "ipd_cm": 99, "focal_mult": -1}, open(p, "w"))
            s, _ = Settings.load(p)
            self.assertEqual(s.diagonal_in, 15.6)     # sayı değil -> varsayılan
            self.assertEqual(s.ipd_cm, 8.0)            # üst sınır
            self.assertEqual(s.focal_mult, 0.2)        # alt sınır


class AspectTest(unittest.TestCase):
    def test_nearest_preset(self):
        self.assertEqual(ASPECT_PRESETS[nearest_aspect_index(1920 / 1080)][0], "16:9")
        self.assertEqual(ASPECT_PRESETS[nearest_aspect_index(1920 / 1200)][0], "16:10")
        self.assertEqual(ASPECT_PRESETS[nearest_aspect_index(1024 / 768)][0], "4:3")


if __name__ == "__main__":
    unittest.main()
