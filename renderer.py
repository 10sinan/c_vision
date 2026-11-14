"""
Renderer - Pygame + PyOpenGL ile off-axis sahne çizimi.
Model yoksa: ekranın arkasına uzanan tel kafes oda ve dönen küp.
Model varsa (--model): dokulu 3B sahne; ekran bir pencere gibi davranır.
"""

import json
import os

import pygame
from pygame.locals import DOUBLEBUF, OPENGL, RESIZABLE
from OpenGL.GL import (
    glClear, glClearColor, glColor4f, glBegin, glEnd, glVertex3f, glLineWidth,
    glMatrixMode, glLoadIdentity, glFrustum, glTranslatef, glRotatef, glScalef,
    glPushMatrix, glPopMatrix, glViewport, glEnable, glDisable, glBlendFunc,
    glLightfv, glLightModeli, glAlphaFunc, glGenLists, glNewList, glEndList,
    glCallList, glGenTextures, glBindTexture, glTexImage2D, glTexParameteri,
    glPixelStorei, glEnableClientState, glDisableClientState, glVertexPointer,
    glNormalPointer, glTexCoordPointer, glColorPointer, glDrawElements,
    GL_COLOR_BUFFER_BIT, GL_DEPTH_BUFFER_BIT, GL_LINES, GL_PROJECTION,
    GL_MODELVIEW, GL_BLEND, GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA, GL_LINE_SMOOTH,
    GL_TRIANGLES, GL_COMPILE, GL_LIGHTING, GL_LIGHT0, GL_COLOR_MATERIAL,
    GL_DEPTH_TEST, GL_POSITION, GL_DIFFUSE, GL_AMBIENT, GL_NORMALIZE,
    GL_LIGHT_MODEL_TWO_SIDE, GL_TRUE, GL_ALPHA_TEST, GL_GREATER, GL_TEXTURE_2D,
    GL_RGBA, GL_UNSIGNED_BYTE, GL_TEXTURE_MIN_FILTER, GL_TEXTURE_MAG_FILTER,
    GL_LINEAR, GL_LINEAR_MIPMAP_LINEAR, GL_TEXTURE_WRAP_S, GL_TEXTURE_WRAP_T,
    GL_REPEAT, GL_GENERATE_MIPMAP, GL_UNPACK_ALIGNMENT, GL_VERTEX_ARRAY,
    GL_NORMAL_ARRAY, GL_TEXTURE_COORD_ARRAY, GL_COLOR_ARRAY, GL_FLOAT,
    GL_UNSIGNED_INT, glClipPlane, GL_CLIP_PLANE0, glDeleteLists, glDeleteTextures,
)

from model_loader import load_model, model_bounds
from off_axis import off_axis_frustum, SCREEN_W_CM, SCREEN_H_CM   # varsayılan ekran boyutu

ROOM_DEPTH_CM = 40.0
GRID_STEPS = 8            # duvarlardaki derinlik çizgisi sayısı
NEAR, FAR = 1.0, 2000.0
MODEL_GAP_CM = 2.0        # modelin ön yüzü ekran camının bu kadar arkasında durur
MAX_TEXTURE_PX = 2048     # bundan büyük dokular küçültülür
# Model başına kaydedilen görünüm ayarları (<model>.view.json) ve varsayılanları
VIEW_DEFAULTS = {"model_scale": 1.0, "cover": True, "depth": 0.0, "yaw": 0.0, "pitch": 0.0}


def _line(a, b):
    glVertex3f(*a)
    glVertex3f(*b)


def _draw_room(screen_w, screen_h):
    w, h, d = screen_w / 2.0, screen_h / 2.0, ROOM_DEPTH_CM
    glBegin(GL_LINES)
    # Duvar kenarları (ekrandan arka duvara)
    glColor4f(0.1, 0.8, 1.0, 1.0)
    for sx in (-1, 1):
        for sy in (-1, 1):
            _line((sx * w, sy * h, 0), (sx * w, sy * h, -d))
    # Arka duvar çerçevesi
    for (x1, y1, x2, y2) in ((-w, -h, w, -h), (w, -h, w, h), (w, h, -w, h), (-w, h, -w, -h)):
        _line((x1, y1, -d), (x2, y2, -d))
    # Derinlik ipuçları: her kademede çerçeve halkası (soluk)
    glColor4f(0.1, 0.8, 1.0, 0.35)
    for i in range(1, GRID_STEPS):
        z = -d * i / GRID_STEPS
        _line((-w, -h, z), (w, -h, z)); _line((w, -h, z), (w, h, z))
        _line((w, h, z), (-w, h, z));   _line((-w, h, z), (-w, -h, z))
    glEnd()


def _draw_cube(center, size):
    s = size / 2.0
    cx, cy, cz = center
    v = [(cx + dx * s, cy + dy * s, cz + dz * s)
         for dz in (1, -1) for dy in (-1, 1) for dx in (-1, 1)]
    # v[0..3] ön yüz, v[4..7] arka yüz
    edges = ((0, 1), (1, 3), (3, 2), (2, 0), (4, 5), (5, 7), (7, 6), (6, 4),
             (0, 4), (1, 5), (2, 6), (3, 7))
    glBegin(GL_LINES)
    glColor4f(1.0, 0.75, 0.15, 1.0)
    for a, b in edges:
        _line(v[a], v[b])
    glEnd()


def _upload_texture(image):
    """PIL görüntüsünü OpenGL dokusuna yükler (mipmap'li). :return: doku kimliği"""
    img = image.convert("RGBA")
    if max(img.size) > MAX_TEXTURE_PX:
        k = MAX_TEXTURE_PX / max(img.size)
        img = img.resize((max(1, int(img.width * k)), max(1, int(img.height * k))))
    # trimesh UV'leri sol-alt kökenli; PIL satırları yukarıdan başlar -> dikey çevir
    data = img.transpose(1).tobytes()  # 1 = FLIP_TOP_BOTTOM
    tex = glGenTextures(1)
    glBindTexture(GL_TEXTURE_2D, tex)
    glPixelStorei(GL_UNPACK_ALIGNMENT, 1)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
    glTexParameteri(GL_TEXTURE_2D, GL_GENERATE_MIPMAP, GL_TRUE)
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, img.width, img.height, 0,
                 GL_RGBA, GL_UNSIGNED_BYTE, data)
    return tex


def _compile_model(parts):
    """Tüm parçaları dokularıyla birlikte tek bir display list'e derler (hızlı çizim).
    :return: (display list kimliği, doku kimlikleri)"""
    # Önce dokular yüklenir; aynı görüntüyü paylaşan parçalar tek doku kullanır
    textures = {}
    for p in parts:
        if p["image"] is not None and id(p["image"]) not in textures:
            textures[id(p["image"])] = _upload_texture(p["image"])

    lst = glGenLists(1)
    glNewList(lst, GL_COMPILE)
    glEnableClientState(GL_VERTEX_ARRAY)
    glEnableClientState(GL_NORMAL_ARRAY)
    for p in parts:
        # Dizi verisi derleme anında listeye kopyalanır
        glVertexPointer(3, GL_FLOAT, 0, p["vertices"])
        glNormalPointer(GL_FLOAT, 0, p["normals"])
        glColor4f(*p["color"])
        textured = p["image"] is not None
        if textured:
            glEnable(GL_TEXTURE_2D)
            glBindTexture(GL_TEXTURE_2D, textures[id(p["image"])])
            glEnableClientState(GL_TEXTURE_COORD_ARRAY)
            glTexCoordPointer(2, GL_FLOAT, 0, p["uv"])
        else:
            glDisable(GL_TEXTURE_2D)
        if p["vcolors"] is not None:
            glEnableClientState(GL_COLOR_ARRAY)
            glColorPointer(4, GL_FLOAT, 0, p["vcolors"])
        glDrawElements(GL_TRIANGLES, p["faces"].size, GL_UNSIGNED_INT, p["faces"])
        if textured:
            glDisableClientState(GL_TEXTURE_COORD_ARRAY)
        if p["vcolors"] is not None:
            glDisableClientState(GL_COLOR_ARRAY)
    glDisable(GL_TEXTURE_2D)
    glDisableClientState(GL_NORMAL_ARRAY)
    glDisableClientState(GL_VERTEX_ARRAY)
    glEndList()
    return lst, list(textures.values())


class Renderer:
    def __init__(self, size=(1280, 720), model_path=None):
        pygame.init()
        pygame.display.set_caption("Parallaks")
        self.size = size
        pygame.display.gl_set_attribute(pygame.GL_DEPTH_SIZE, 24)
        pygame.display.set_mode(size, DOUBLEBUF | OPENGL | RESIZABLE)
        glClearColor(0.02, 0.02, 0.06, 1.0)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glEnable(GL_LINE_SMOOTH)
        glLineWidth(1.8)
        self.angle = 0.0
        self.screen_w, self.screen_h = SCREEN_W_CM, SCREEN_H_CM   # fiziksel ekran (cm); set_screen() ile değişir
        self._ext = None            # yüklü modelin boyutları (ölçek hesabı için)

        self.model_list = None
        self.model_path = None
        self._textures = []
        self.show_room = True
        # Görünüm ayarları (VIEW_DEFAULTS):
        #   model_scale  boyut çarpanı
        #   cover        True: ekranı boşluksuz doldur (kenarlar kesilir), False: sığdır
        #   depth        sahneyi ekrana doğru (+) / uzağa (-) kaydırma, cm
        #   yaw          sahneyi dikey eksen etrafında döndürme, derece
        #   pitch        sahneyi öne/arkaya eğme, derece (+: üstten bakış)
        self.reset_view()
        self._view_path = None
        if model_path:
            self.load_model(model_path)

    def load_model(self, model_path):
        """Modeli yükler; varsa önceki modelin görünümünü kaydedip GPU kaynaklarını bırakır.
        model_path=None: modelsiz moda (tel kafes oda + küp) döner."""
        self.save_view()
        self.unload_model()
        self.reset_view()
        if model_path is None:
            self.show_room = True
            return
        parts = load_model(model_path)
        lo, hi = model_bounds(parts)
        self._center = (lo + hi) / 2.0
        self._half_depth = float(hi[2] - lo[2]) / 2.0
        self._ext = [max(float(e), 1e-9) for e in (hi - lo)]
        self._update_scales()
        self.model_list, self._textures = _compile_model(parts)
        self.model_path = model_path
        self._view_path = os.path.splitext(model_path)[0] + ".view.json"
        self._load_view()
        self.show_room = False  # model varken tel kafes çizgiler gizli (L ile açılır)

    def set_screen(self, width_cm, height_cm):
        """Fiziksel ekran boyutunu ayarlar (cm); oda çizgileri, model ölçeği ve frustum buna uyar."""
        self.screen_w, self.screen_h = float(width_cm), float(height_cm)
        self._update_scales()

    def _update_scales(self):
        if self._ext is None:
            return
        ex, ey = self._ext[0], self._ext[1]
        self._fit_scale = min(self.screen_w / ex, self.screen_h / ey)
        self._cover_scale = max(self.screen_w / ex, self.screen_h / ey)

    def unload_model(self):
        if self.model_list is not None:
            glDeleteLists(self.model_list, 1)
        if self._textures:
            glDeleteTextures(self._textures)
        self.model_list, self._textures = None, []
        self.model_path = self._view_path = None
        self._ext = None

    def reset_view(self):
        for k, v in VIEW_DEFAULTS.items():
            setattr(self, k, v)

    def _load_view(self):
        try:
            with open(self._view_path) as fh:
                saved = json.load(fh)
        except (OSError, ValueError):
            return
        for k in VIEW_DEFAULTS:
            if k in saved:
                setattr(self, k, type(VIEW_DEFAULTS[k])(saved[k]))

    def save_view(self):
        """Model görünüm ayarlarını <model>.view.json dosyasına yazar (bir dahaki açılışta yüklenir)."""
        if self._view_path is None:
            return
        with open(self._view_path, "w") as fh:
            json.dump({k: getattr(self, k) for k in VIEW_DEFAULTS}, fh, indent=2)

    def resize(self, size):
        self.size = size
        glViewport(0, 0, *size)

    def _draw_model(self):
        s = (self._cover_scale if self.cover else self._fit_scale) * self.model_scale
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_LIGHTING)
        glEnable(GL_LIGHT0)
        glEnable(GL_NORMALIZE)
        glEnable(GL_COLOR_MATERIAL)
        glLightModeli(GL_LIGHT_MODEL_TWO_SIDE, GL_TRUE)   # yaprak/çimen gibi tek yüzlü düzlemler
        glEnable(GL_ALPHA_TEST)                          # dokudaki saydam pikselleri at
        glAlphaFunc(GL_GREATER, 0.4)
        # Işık dünyaya sabit (kamera ötelemesinden sonra, model dönüşümünden önce)
        glLightfv(GL_LIGHT0, GL_POSITION, (0.3, 0.6, 1.0, 0.0))
        glLightfv(GL_LIGHT0, GL_DIFFUSE, (0.75, 0.75, 0.75, 1.0))
        glLightfv(GL_LIGHT0, GL_AMBIENT, (0.45, 0.45, 0.5, 1.0))

        # Ekran camının önünde kalan her şey kesilir (gerçek pencere gibi): z <= 0 tutulur.
        # Düzlem şu anki (dünya) koordinatlarında tanımlanır.
        glClipPlane(GL_CLIP_PLANE0, (0.0, 0.0, -1.0, 0.0))
        glEnable(GL_CLIP_PLANE0)

        glPushMatrix()
        # Ön yüz ekran camının MODEL_GAP_CM arkasında, sahne ekranın ortasında
        glTranslatef(0.0, 0.0, -MODEL_GAP_CM - self._half_depth * s + self.depth)
        glRotatef(self.pitch, 1.0, 0.0, 0.0)
        glRotatef(self.yaw, 0.0, 1.0, 0.0)
        glScalef(s, s, s)
        glTranslatef(*(-self._center))
        glCallList(self.model_list)
        glPopMatrix()

        glDisable(GL_CLIP_PLANE0)
        glDisable(GL_ALPHA_TEST)
        glDisable(GL_LIGHTING)
        glDisable(GL_DEPTH_TEST)

    def draw(self, eye, dt=0.0):
        """eye: ekran merkezine göre (ex, ey, ez) cm."""
        self.angle += 25.0 * dt
        l, r, b, t, n, f = off_axis_frustum(eye, NEAR, FAR, self.screen_w, self.screen_h)

        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        glFrustum(l, r, b, t, n, f)

        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()
        glTranslatef(-eye[0], -eye[1], -eye[2])   # kamera ötelenir, DÖNMEZ

        if self.show_room:
            _draw_room(self.screen_w, self.screen_h)
        if self.model_list is not None:
            self._draw_model()
        else:
            glPushMatrix()
            glTranslatef(0.0, 0.0, -ROOM_DEPTH_CM * 0.5)
            glRotatef(self.angle, 0.3, 1.0, 0.0)
            _draw_cube((0.0, 0.0, 0.0), 8.0)
            glPopMatrix()

        pygame.display.flip()
