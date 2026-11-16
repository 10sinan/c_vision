"""panel.py yerleşim/tıklama testleri (OpenGL ve pencere gerektirmez). Çalıştır: python -m unittest test_panel -v"""

import unittest

import panel as P

WIN = (1280, 720)


def make(n_extra=0):
    log = []
    state = {"on": False, "enabled": True}
    rows = [
        P.Row("Derinlik", "step", lambda d: log.append(("step", d)), lambda: "0 cm"),
        P.Row("Efektler", "toggle", lambda d: state.update(on=not state["on"]), lambda: state["on"]),
        P.Row("Menü", "action", lambda d: log.append(("action", d))),
        P.Row("Kapalı satır", "step", lambda d: log.append(("disabled", d)), lambda: "-",
              enabled=lambda: state["enabled"]),
    ] + [P.Row(f"Ek {i}", "action", lambda d: None) for i in range(n_extra)]
    return P.Panel(rows), log, state


def row_point(p, i, where):
    """i. satırda bir nokta: where = 'minus' | 'plus' | 'label'"""
    x, y, w, _ = p.rect(WIN)
    cy = y + P.HEADER_H + i * p._row_h(WIN) + p._row_h(WIN) // 2
    cx = {"plus": x + w - P.PAD - P.BTN_W // 2,
          "minus": x + w - P.PAD - P.BTN_W - 4 - P.BTN_W // 2,
          "label": x + P.PAD + 5}[where]
    return cx, cy


class PanelTest(unittest.TestCase):
    def test_closed_panel_only_reacts_on_tab_button(self):
        p, log, _ = make()
        self.assertFalse(p.click((100, 300), WIN))            # sahne: panele ait değil
        self.assertFalse(p.open)
        x, y, w, h = p.rect(WIN)
        self.assertEqual((w, h), (P.TAB_W, P.TAB_H))
        self.assertTrue(p.click((x + 5, y + 5), WIN))
        self.assertTrue(p.open)

    def test_close_button(self):
        p, _, _ = make()
        p.open = True
        x, y, w, _ = p.rect(WIN)
        self.assertEqual(p.hit((x + w - P.PAD - 5, y + 20), WIN), ("close",))
        self.assertTrue(p.click((x + w - P.PAD - 5, y + 20), WIN))
        self.assertFalse(p.open)

    def test_step_buttons_and_label_area(self):
        p, log, _ = make()
        p.open = True
        self.assertTrue(p.click(row_point(p, 0, "plus"), WIN))
        self.assertTrue(p.click(row_point(p, 0, "minus"), WIN))
        self.assertTrue(p.click(row_point(p, 0, "label"), WIN))   # panele ait ama bir şey yapmaz
        self.assertEqual(log, [("step", 1), ("step", -1)])

    def test_toggle_and_action_use_whole_row(self):
        p, log, state = make()
        p.open = True
        p.click(row_point(p, 1, "label"), WIN)
        self.assertTrue(state["on"])
        p.click(row_point(p, 1, "plus"), WIN)
        self.assertFalse(state["on"])
        p.click(row_point(p, 2, "label"), WIN)
        self.assertEqual(log, [("action", 1)])

    def test_disabled_row_ignores_clicks_but_consumes_them(self):
        p, log, state = make()
        p.open = True
        state["enabled"] = False
        self.assertTrue(p.click(row_point(p, 3, "plus"), WIN))
        self.assertTrue(p.wheel(row_point(p, 3, "label"), WIN, 1))
        self.assertEqual(log, [])

    def test_wheel_changes_step_rows_only(self):
        p, log, state = make()
        p.open = True
        self.assertTrue(p.wheel(row_point(p, 0, "label"), WIN, 1))
        self.assertTrue(p.wheel(row_point(p, 0, "label"), WIN, -2))
        self.assertTrue(p.wheel(row_point(p, 1, "label"), WIN, 1))   # toggle: tekerlek etkisiz
        self.assertFalse(p.wheel((100, 300), WIN, 1))                # panel dışı
        self.assertEqual(log, [("step", 1), ("step", -1)])
        self.assertFalse(state["on"])

    def test_panel_fits_short_window(self):
        p, _, _ = make(n_extra=16)                    # 20 satır
        p.open = True
        small = (800, 500)
        x, y, w, h = p.rect(small)
        self.assertLessEqual(y + h, small[1])
        self.assertGreaterEqual(x, 0)
        self.assertEqual(p.rect((1280, 900))[3], P.HEADER_H + 20 * P.ROW_H + P.PAD)   # yeterince yüksekte tam boy

    def test_outside_and_none_position(self):
        p, _, _ = make()
        p.open = True
        self.assertIsNone(p.hit(None, WIN))
        self.assertIsNone(p.hit((10, 10), WIN))


if __name__ == "__main__":
    unittest.main()
