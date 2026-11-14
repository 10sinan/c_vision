"""thumbnails.py için testler (OpenGL gerektirmez; renderer taklit edilir).
Çalıştır: python -m unittest test_thumbnails -v"""

import os
import tempfile
import unittest

import numpy as np

import thumbnails as T


def _write(path, data):
    with open(path, "wb") as fh:
        fh.write(data)


class FakeRenderer:
    """ensure_thumbnail'in kullandığı arayüz: load_model, draw, grab_frame."""

    def __init__(self):
        self.loaded = []
        self.size = (1280, 720)

    def load_model(self, path):
        self.loaded.append(path)

    def draw(self, eye, dt=0.0, present=True):
        assert present is False, "önizleme ekrana basılmamalı"

    def grab_frame(self):
        w, h = self.size
        frame = np.zeros((h, w, 3), dtype=np.uint8)
        frame[:, :, 0] = 200   # kırmızı kanal
        return frame


class ThumbPathTest(unittest.TestCase):
    def test_path_changes_when_file_changes(self):
        with tempfile.TemporaryDirectory() as d:
            model = os.path.join(d, "araba.glb")
            _write(model, b"x" * 10)
            os.utime(model, (1000, 1000))
            a = T.thumb_path(model, d)
            _write(model, b"x" * 20)        # boyut değişti
            os.utime(model, (1000, 1000))
            b = T.thumb_path(model, d)
            os.utime(model, (2000, 2000))              # mtime değişti
            c = T.thumb_path(model, d)
            self.assertEqual(len({a, b, c}), 3)

    def test_wireframe_has_fixed_path(self):
        self.assertTrue(T.thumb_path(None, "/x").endswith(T.WIREFRAME_KEY + ".png"))


class EnsureThumbnailTest(unittest.TestCase):
    def test_generates_once_and_caches(self):
        with tempfile.TemporaryDirectory() as d:
            model = os.path.join(d, "m.glb")
            _write(model, b"data")
            tdir = os.path.join(d, "thumbs")
            r = FakeRenderer()
            p1 = T.ensure_thumbnail(r, model, tdir)
            p2 = T.ensure_thumbnail(r, model, tdir)
            self.assertEqual(p1, p2)
            self.assertEqual(r.loaded, [model])          # ikinci çağrıda yeniden çizilmedi
            data, w, h = T.load_thumbnail_rgb(p1, 160)
            self.assertEqual((w, h), (160, 90))
            self.assertEqual(len(data), 160 * 90 * 3)

    def test_stale_thumbnails_are_removed(self):
        with tempfile.TemporaryDirectory() as d:
            model = os.path.join(d, "m.glb")
            _write(model, b"data")
            os.utime(model, (1000, 1000))
            tdir = os.path.join(d, "thumbs")
            r = FakeRenderer()
            old = T.ensure_thumbnail(r, model, tdir)
            _write(model, b"newer data")
            new = T.ensure_thumbnail(r, model, tdir)
            self.assertNotEqual(old, new)
            self.assertFalse(os.path.exists(old))
            self.assertTrue(os.path.exists(new))

    def test_prune_does_not_touch_models_with_prefix_names(self):
        with tempfile.TemporaryDirectory() as d:
            tdir = os.path.join(d, "thumbs")
            os.makedirs(tdir)
            other = os.path.join(tdir, "araba-eski-1000-5.png")   # başka modelin adı "araba-eski"
            _write(other, b"png")
            model = os.path.join(d, "araba.glb")
            _write(model, b"d")
            T.ensure_thumbnail(FakeRenderer(), model, tdir)
            self.assertTrue(os.path.exists(other))

    def test_missing_thumbnail_file_returns_none(self):
        self.assertIsNone(T.load_thumbnail_rgb("/nonexistent.png"))


if __name__ == "__main__":
    unittest.main()
