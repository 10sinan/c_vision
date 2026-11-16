"""off_axis.py için küçük testler (OpenGL gerektirmez). Çalıştır: python -m unittest test_off_axis -v"""

import unittest

from off_axis import (head_to_eye_cm, off_axis_frustum,
                      SCREEN_W_CM, SCREEN_H_CM, CAMERA_OFFSET_Y_CM)


class HeadToEyeTest(unittest.TestCase):
    def test_center_pose_sits_at_camera_offset(self):
        ex, ey, ez = head_to_eye_cm(0.0, 0.0, 60.0, 640, 480, 544.0)
        self.assertAlmostEqual(ex, 0.0)
        self.assertAlmostEqual(ey, CAMERA_OFFSET_Y_CM)
        self.assertAlmostEqual(ez, 60.0)

    def test_right_of_frame_is_positive_x(self):
        ex, _, _ = head_to_eye_cm(1.0, 0.0, 60.0, 640, 480, 544.0)
        self.assertGreater(ex, 0.0)


class AmplifyEyeTest(unittest.TestCase):
    def test_gain_one_changes_nothing(self):
        from off_axis import amplify_eye
        self.assertEqual(amplify_eye((3.0, 12.0, 55.0), 1.0, 10.7), (3.0, 12.0, 55.0))

    def test_scales_offsets_around_neutral_point_and_keeps_distance(self):
        from off_axis import amplify_eye
        ex, ey, ez = amplify_eye((4.0, 12.7, 55.0), 2.5, 10.7)
        self.assertAlmostEqual(ex, 10.0)
        self.assertAlmostEqual(ey, 10.7 + 2.0 * 2.5)
        self.assertEqual(ez, 55.0)
        self.assertEqual(amplify_eye((0.0, 10.7, 60.0), 4.0, 10.7), (0.0, 10.7, 60.0))   # nötr nokta sabit


class FrustumTest(unittest.TestCase):
    def test_centered_eye_gives_symmetric_frustum(self):
        left, right, bottom, top, near, far = off_axis_frustum((0.0, 0.0, 60.0), 1.0, 300.0)
        self.assertAlmostEqual(left, -right)
        self.assertAlmostEqual(bottom, -top)
        self.assertAlmostEqual(right, SCREEN_W_CM / 2 / 60.0)
        self.assertAlmostEqual(top, SCREEN_H_CM / 2 / 60.0)

    def test_eye_moving_right_shifts_frustum_left(self):
        left, right, *_ = off_axis_frustum((10.0, 0.0, 60.0), 1.0, 300.0)
        self.assertLess(left, -right)  # sol kenar daha uzak, sağ kenar daha yakın

    def test_zero_depth_does_not_crash(self):
        off_axis_frustum((0.0, 0.0, 0.0), 1.0, 300.0)


if __name__ == "__main__":
    unittest.main()
