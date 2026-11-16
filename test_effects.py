"""effects.py için testler (OpenGL gerektirmez). Çalıştır: python -m unittest test_effects -v"""

import unittest

import numpy as np

import effects as E

W, H = 34.5, 19.4


class BoundsTest(unittest.TestCase):
    def test_foreground_objects_stay_in_front_and_visible(self):
        fx = E.Effects(W, H)
        for t in np.linspace(0, 600, 400):             # 10 dakikalık simülasyon
            fx.t = float(t)
            for pos in (fx.leaf_positions(),):
                z_lo = E.FG_Z_RANGE[0] - 3
                z_hi = E.FG_Z_RANGE[1] + 3
                self.assertTrue(((pos[:, 2] > z_lo) & (pos[:, 2] < z_hi)).all())
                # yatay/dikey: ekran penceresinde (marj + salınım payıyla) kalır
                xh = E.visible_half_extent(W / 2, E.FG_Z_RANGE[0]) + 6
                yh = E.visible_half_extent(H / 2, E.FG_Z_RANGE[0]) + 6
                self.assertTrue((np.abs(pos[:, 0]) < xh).all())
                self.assertTrue((np.abs(pos[:, 1]) < yh).all())

    def test_dust_is_bounded(self):
        fx = E.Effects(W, H)
        for t in np.linspace(0, 600, 200):
            fx.t = float(t)
            p = fx.dust_positions()
            self.assertTrue(((p[:, 2] > E.DUST_Z_RANGE[0] - 1) & (p[:, 2] < E.DUST_Z_RANGE[1] + 1)).all())
            self.assertTrue(np.isfinite(p).all())

    def test_leaves_actually_fall(self):
        fx = E.Effects(W, H)
        y0 = fx.leaf_positions()[:, 1].copy()
        fx.update(0.5)
        y1 = fx.leaf_positions()[:, 1]
        moved_down = (y1 < y0) | (y1 > y0 + 5)          # aşağı indi ya da üstten sarıldı
        self.assertTrue(moved_down.all())


class TimeTest(unittest.TestCase):
    def test_zero_dt_changes_nothing(self):
        fx = E.Effects(W, H)
        a = fx.dust_positions().copy(), fx.leaf_positions().copy()
        fx.update(0.0)
        self.assertTrue(np.array_equal(a[0], fx.dust_positions()))
        self.assertTrue(np.array_equal(a[1], fx.leaf_positions()))

    def test_positions_depend_on_total_time_not_step_size(self):
        a, b = E.Effects(W, H), E.Effects(W, H)
        for _ in range(100):
            a.update(0.01)
        b.update(1.0)
        self.assertTrue(np.allclose(a.leaf_positions(), b.leaf_positions()))

    def test_negative_dt_is_ignored(self):
        fx = E.Effects(W, H)
        fx.update(-5.0)
        self.assertEqual(fx.t, 0.0)


class SetupTest(unittest.TestCase):
    def test_deterministic_for_same_seed(self):
        a, b = E.Effects(W, H, seed=3), E.Effects(W, H, seed=3)
        self.assertTrue(np.array_equal(a.dust_positions(), b.dust_positions()))
        c = E.Effects(W, H, seed=4)
        self.assertFalse(np.array_equal(a.dust_positions(), c.dust_positions()))

    def test_density_scales_counts(self):
        full, half, none = E.Effects(W, H, 1.0), E.Effects(W, H, 0.5), E.Effects(W, H, 0.0)
        self.assertEqual(none.counts, (0, 0))
        for f, h in zip(full.counts, half.counts):
            self.assertLess(h, f)
        half.set_density(1.0)
        self.assertEqual(half.counts, full.counts)

    def test_empty_effects_do_not_crash(self):
        fx = E.Effects(W, H, 0.0)
        fx.update(1.0)
        self.assertEqual(fx.dust_positions().shape, (0, 3))
        self.assertEqual(fx.leaf_positions().shape, (0, 3))


class KindTest(unittest.TestCase):
    def test_only_selected_kind_produces_particles(self):
        fx = E.Effects(W, H)
        sizes = {}
        for kind in range(len(E.KIND_NAMES)):
            fx.set_kind(kind)
            sizes[kind] = (sum(fx.counts), len(fx.rain_segments()), len(fx.snow_positions()),
                           len(fx.firefly_points()[0]))
        for kind, row in sizes.items():
            self.assertEqual([n > 0 for n in row], [i == kind for i in range(4)])

    def test_kind_wraps_and_does_not_disturb_leaves(self):
        a, b = E.Effects(W, H), E.Effects(W, H)
        b.set_kind(E.RAIN)
        b.set_kind(len(E.KIND_NAMES))                   # sarar -> yaprak
        self.assertEqual(b.kind, E.LEAVES)
        self.assertTrue(np.array_equal(a.leaf_positions(), b.leaf_positions()))

    def test_weather_stays_bounded_and_finite(self):
        fx = E.Effects(W, H)
        z_lo, z_hi = E.DUST_Z_RANGE[0] - 8, E.DUST_Z_RANGE[1] + 8
        x_max = (W / 2) * (E.REF_EYE_Z - E.DUST_Z_RANGE[0]) / E.REF_EYE_Z * E.VIEW_MARGIN + 8
        for t in np.linspace(0, 600, 150):
            fx.t = float(t)
            fx.set_kind(E.RAIN)
            rain = fx.rain_segments().reshape(-1, 3)
            fx.set_kind(E.SNOW)
            snow = fx.snow_positions()
            fx.set_kind(E.FIREFLIES)
            flies, glow = fx.firefly_points()
            for p in (rain, snow, flies):
                self.assertTrue(np.isfinite(p).all())
                self.assertTrue(((p[:, 2] > z_lo) & (p[:, 2] < z_hi)).all())
                self.assertTrue((np.abs(p[:, 0]) < x_max).all())
            self.assertTrue(((glow >= 0) & (glow <= 1)).all())

    def test_rain_and_snow_fall(self):
        fx = E.Effects(W, H, kind=E.RAIN)
        seg = fx.rain_segments()
        self.assertTrue((seg[:, 0, 1] < seg[:, 1, 1]).all())            # baş (alt uç) kuyruğun altında
        self.assertTrue(np.allclose(np.linalg.norm(seg[:, 0] - seg[:, 1], axis=1), fx._rain_len))
        y0 = seg[:, 0, 1].copy()
        fx.update(0.02)
        y1 = fx.rain_segments()[:, 0, 1]
        self.assertGreater(np.mean((y1 < y0)), 0.9)                     # sarılanlar dışında hepsi indi
        fx.set_kind(E.SNOW)
        y0 = fx.snow_positions()[:, 1].copy()
        fx.update(0.1)
        self.assertGreater(np.mean(fx.snow_positions()[:, 1] < y0), 0.9)

    def test_fireflies_blink(self):
        fx = E.Effects(W, H, kind=E.FIREFLIES)
        glows = []
        for t in np.linspace(0, 10, 200):
            fx.t = float(t)
            glows.append(fx.firefly_points()[1])
        glows = np.array(glows)
        self.assertTrue((glows.max(axis=0) > 0.8).all())                # her biri parlıyor
        self.assertTrue((glows.min(axis=0) == 0.0).all())               # ve tamamen sönüyor


class ShapeTest(unittest.TestCase):
    def test_leaf_keeps_its_size_and_color_range(self):
        fx = E.Effects(W, H)
        for t in (0.0, 1.3, 7.9):
            fx.t = t
            verts, color = fx.leaf_quad(0)
            self.assertEqual(verts.shape, (4, 3))
            length = np.linalg.norm(verts[0] - verts[2])        # elmasın uzun köşegeni = 2*boyut (döndürme korur)
            self.assertAlmostEqual(length, 2 * fx._leaf_size[0], places=6)
            self.assertTrue(((color >= 0) & (color <= 1)).all())


if __name__ == "__main__":
    unittest.main()
