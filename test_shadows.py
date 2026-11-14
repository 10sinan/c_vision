"""shadows.py'nin saf matematiği için testler (OpenGL gerektirmez).
Çalıştır: python -m unittest test_shadows -v"""

import unittest

import numpy as np

import shadows as S


def to_clip(V, P, pts):
    h = np.hstack([pts, np.ones((len(pts), 1))])
    return (P @ V @ h.T).T


class LightMatricesTest(unittest.TestCase):
    def _sphere_points(self, center, radius, n=500, seed=0):
        rng = np.random.default_rng(seed)
        v = rng.normal(size=(n, 3))
        v /= np.linalg.norm(v, axis=1, keepdims=True)
        return center + v * radius * rng.random((n, 1)) ** (1 / 3)

    def test_every_point_inside_the_sphere_is_inside_the_light_frustum(self):
        for center, radius, ldir in (((0, 0, -30), 50.0, (0.4, 0.9, 0.5)),
                                     ((5, -2, -12), 3.0, (0.0, 1.0, 0.0)),     # ışık tam yukarıdan
                                     ((0, 0, 0), 1.0, (0.0, 0.0, 1.0))):        # ışık tam önden
            V, P = S.light_matrices(center, radius, ldir)
            clip = to_clip(V, P, self._sphere_points(np.array(center, float), radius))
            self.assertTrue((np.abs(clip[:, :3]) <= 1.0 + 1e-9).all(), (center, radius, ldir))

    def test_sphere_center_maps_to_clip_origin_xy(self):
        V, P = S.light_matrices((3, 4, -20), 10.0, (0.4, 0.9, 0.5))
        c = to_clip(V, P, np.array([[3.0, 4.0, -20.0]]))[0]
        self.assertAlmostEqual(c[0], 0.0)
        self.assertAlmostEqual(c[1], 0.0)

    def test_closer_to_light_means_smaller_depth(self):
        center = np.array([0.0, 0.0, -20.0])
        ldir = np.array([0.4, 0.9, 0.5]) / np.linalg.norm([0.4, 0.9, 0.5])
        V, P = S.light_matrices(center, 10.0, ldir)
        near_light = to_clip(V, P, (center + ldir * 5)[None])[0, 2]
        far_light = to_clip(V, P, (center - ldir * 5)[None])[0, 2]
        self.assertLess(near_light, far_light)

    def test_view_matrix_is_rigid(self):
        V = S.look_at((1, 2, 3), (0, 0, 0), (0, 0, 1))
        R = V[:3, :3]
        self.assertTrue(np.allclose(R @ R.T, np.eye(3)))
        self.assertAlmostEqual(np.linalg.det(R), 1.0)

    def test_parallel_up_vector_does_not_break(self):
        V = S.look_at((0, 0, 10), (0, 0, 0), (0, 0, 1))     # bakış yönü up ile paralel
        self.assertTrue(np.isfinite(V).all())
        self.assertTrue(np.allclose(V[:3, :3] @ V[:3, :3].T, np.eye(3)))


class ShadowsDisabledTest(unittest.TestCase):
    def test_env_var_disables_without_gl(self):
        import os
        os.environ["PARALLAX_NO_SHADOWS"] = "1"
        try:
            sh = S.Shadows()
            self.assertFalse(sh.available)
        finally:
            del os.environ["PARALLAX_NO_SHADOWS"]


if __name__ == "__main__":
    unittest.main()
