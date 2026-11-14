"""recorder.py için testler (OpenGL gerektirmez). Çalıştır: python -m unittest test_recorder -v"""

import os
import tempfile
import time
import unittest

import cv2
import numpy as np

import recorder as R


def scene(color=(255, 0, 0), size=(720, 1280)):
    img = np.zeros((size[0], size[1], 3), dtype=np.uint8)
    img[:] = color
    return img


class ComposeTest(unittest.TestCase):
    def test_output_shape_and_layout(self):
        pose = {"left_eye_px": (200, 200), "right_eye_px": (260, 200)}
        cam = np.zeros((480, 640, 3), dtype=np.uint8)
        out = R.compose_frame(scene((255, 0, 0)), cam, pose, 61.0)
        self.assertEqual(out.shape, (R.OUT_H, R.OUT_W, 3))
        # RGB kırmızı sahne, BGR çıktıda (0,0,255) olmalı; sol panelin ortası
        self.assertEqual(tuple(out[R.PANEL_H // 2, R.SCENE_W // 2]), (0, 0, 255))
        # sağ panelde yeşil işaretler çizildi
        self.assertTrue((out[:, R.SCENE_W:, 1] == 255).any())

    def test_no_markers_without_pose(self):
        cam = np.zeros((480, 640, 3), dtype=np.uint8)
        out = R.compose_frame(scene(), cam, None, None)
        self.assertFalse(out[:, R.SCENE_W:].any())

    def test_mouse_mode_has_text_panel(self):
        out = R.compose_frame(scene(), None)
        self.assertEqual(out.shape, (R.OUT_H, R.OUT_W, 3))
        self.assertTrue(out[:, R.SCENE_W:].any())     # "mouse mode" yazısı

    def test_non_16_9_scene_is_letterboxed_not_stretched(self):
        out = R.compose_frame(scene((0, 255, 0), size=(800, 800)), None)   # kare sahne
        self.assertEqual(tuple(out[10, 10]), (0, 0, 0))                    # sol üst köşe: siyah bant
        self.assertEqual(tuple(out[R.PANEL_H // 2, R.SCENE_W // 2]), (0, 255, 0))


class RecorderTest(unittest.TestCase):
    def test_records_real_time_duration(self):
        with tempfile.TemporaryDirectory() as d:
            rec = R.Recorder(d)
            path = rec.start()
            frame = scene()
            t_end = time.monotonic() + 1.0
            while time.monotonic() < t_end:
                if rec.due():
                    rec.capture(frame, None)
                time.sleep(0.002)
            self.assertEqual(rec.stop(), path)
            cap = cv2.VideoCapture(path)
            n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            self.assertEqual((int(cap.get(3)), int(cap.get(4))), (R.OUT_W, R.OUT_H))
            self.assertAlmostEqual(n / R.FPS, 1.0, delta=0.25)       # ~1 saniyelik video

    def test_slow_capture_duplicates_frames_to_keep_duration(self):
        with tempfile.TemporaryDirectory() as d:
            rec = R.Recorder(d)
            rec.start()
            frame = scene()
            for _ in range(4):                      # saniyede ~4 kare yakalayan yavaş döngü
                rec.capture(frame, None)
                time.sleep(0.25)
            rec.stop()
            self.assertAlmostEqual(rec.written / R.FPS, 0.75, delta=0.25)

    def test_full_queue_drops_frames_without_blocking(self):
        with tempfile.TemporaryDirectory() as d:
            rec = R.Recorder(d)
            rec.start()
            rec._queue.maxsize = 1
            frame = scene()                                    # kareyi döngü dışında üret (ölçüm yalnızca capture'ı kapsasın)
            t0 = time.monotonic()
            for _ in range(200):
                rec.capture(frame, None)
            self.assertLess(time.monotonic() - t0, 0.2)        # capture asla beklemez
            rec.stop()

    def test_stop_when_not_active_is_noop(self):
        self.assertIsNone(R.Recorder("/tmp/x").stop())


if __name__ == "__main__":
    unittest.main()
