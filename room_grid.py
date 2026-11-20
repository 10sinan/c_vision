"""
Oda duvarlarındaki ızgara çizgileri (derinlik ipucu). OpenGL'siz, test edilebilir.

Düz renkli duvarlarda göz perspektifi yakalayamaz; düzenli aralıklı çizgiler, baş oynayınca
yakın/uzak farkını net gösterir.
"""

GRID_STEP_CM = 4.0       # yaklaşık çizgi aralığı (duvar boyuna tam bölünecek şekilde ayarlanır)
LINE_OFFSET_CM = 0.04    # çizgiler duvarın biraz önünde durur (z-çakışmasın)


def _divisions(length, step):
    return max(1, round(length / step))


def wall_segments(origin, u_vec, v_vec, normal, step=GRID_STEP_CM, offset=LINE_OFFSET_CM):
    """
    Bir duvarın iç ızgara çizgileri (kenar çizgileri hariç; onlar köşede zaten görünür).
    :param origin: duvarın köşesi; u_vec / v_vec: duvarın iki kenar vektörü; normal: içe bakan birim normal
    :return: [(p0, p1), ...] 3'lü koordinat çiftleri
    """
    def length(v):
        return sum(c * c for c in v) ** 0.5

    nu, nv = _divisions(length(u_vec), step), _divisions(length(v_vec), step)
    base = tuple(origin[i] + normal[i] * offset for i in range(3))

    def point(u, v):
        return tuple(base[i] + u_vec[i] * u + v_vec[i] * v for i in range(3))

    segs = []
    for i in range(1, nu):                    # v yönünde uzanan çizgiler
        u = i / nu
        segs.append((point(u, 0.0), point(u, 1.0)))
    for j in range(1, nv):                    # u yönünde uzanan çizgiler
        v = j / nv
        segs.append((point(0.0, v), point(1.0, v)))
    return segs


def room_segments(walls, step=GRID_STEP_CM):
    """:param walls: [(köşe, u_vec, v_vec, normal), ...]  :return: tüm duvarların çizgileri"""
    out = []
    for origin, u_vec, v_vec, normal in walls:
        out.extend(wall_segments(origin, u_vec, v_vec, normal, step))
    return out
