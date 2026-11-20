import unittest
from room_grid import wall_segments, room_segments, LINE_OFFSET_CM
from menu import guide_lines


class RoomGridTest(unittest.TestCase):
    def test_line_count_matches_divisions(self):
        # 20 x 8 cm duvar, 4 cm aralık -> u'da 5 bölme (4 iç çizgi), v'de 2 bölme (1 iç çizgi)
        segs = wall_segments((0, 0, 0), (20, 0, 0), (0, 8, 0), (0, 0, 1), step=4.0)
        self.assertEqual(len(segs), 4 + 1)

    def test_lines_lie_inside_wall_and_offset_along_normal(self):
        segs = wall_segments((0, 0, 0), (20, 0, 0), (0, 8, 0), (0, 0, 1), step=4.0)
        for p0, p1 in segs:
            for p in (p0, p1):
                self.assertAlmostEqual(p[2], LINE_OFFSET_CM)
                self.assertTrue(0.0 <= p[0] <= 20.0 and 0.0 <= p[1] <= 8.0)

    def test_tiny_wall_has_no_interior_lines(self):
        self.assertEqual(wall_segments((0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1)), [])

    def test_room_segments_concatenates_walls(self):
        w = ((0, 0, 0), (20, 0, 0), (0, 8, 0), (0, 0, 1))
        self.assertEqual(len(room_segments([w, w], step=4.0)), 10)


class GuideTest(unittest.TestCase):
    def test_guide_differs_by_camera(self):
        self.assertNotEqual(guide_lines(True), guide_lines(False))
        self.assertTrue(all(h and b for h, b in guide_lines(True) + guide_lines(False)))


if __name__ == "__main__":
    unittest.main()
