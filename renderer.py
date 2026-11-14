"""
Renderer (Aşama 3) - Pygame + PyOpenGL ile tel kafes oda ve küp.
Oda, ön yüzü fiziksel ekranla çakışacak şekilde ekranın arkasına uzanır.
"""

import math
import pygame
from pygame.locals import DOUBLEBUF, OPENGL, RESIZABLE
from OpenGL.GL import (
    glClear, glClearColor, glColor4f, glBegin, glEnd, glVertex3f, glLineWidth,
    glMatrixMode, glLoadIdentity, glFrustum, glTranslatef, glRotatef,
    glPushMatrix, glPopMatrix, glViewport, glEnable, glBlendFunc,
    GL_COLOR_BUFFER_BIT, GL_DEPTH_BUFFER_BIT, GL_LINES, GL_PROJECTION,
    GL_MODELVIEW, GL_BLEND, GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA,
    GL_LINE_SMOOTH, GL_TRIANGLES, GL_COMPILE, GL_LIGHTING, GL_LIGHT0,
    GL_COLOR_MATERIAL, GL_DEPTH_TEST, GL_POSITION, GL_DIFFUSE, GL_AMBIENT,
    GL_NORMALIZE, glNormal3f, glGenLists, glNewList, glEndList, glCallList,
    glDisable, glLightfv, glColor3f, glScalef,
)

from obj_loader import load_model, fit_to_size
from off_axis import off_axis_frustum, SCREEN_W_CM, SCREEN_H_CM

ROOM_DEPTH_CM = 40.0
GRID_STEPS = 8          # duvarlardaki derinlik çizgisi sayısı
NEAR, FAR = 1.0, 300.0
MODEL_FILL = 1.0         # model ekranın (genişlik/yükseklik) bu oranına sığdırılır


def _line(a, b):
    glVertex3f(*a)
    glVertex3f(*b)


def _draw_room():
    w, h, d = SCREEN_W_CM / 2.0, SCREEN_H_CM / 2.0, ROOM_DEPTH_CM
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


def _compile_model(path):
    """OBJ dosyasını yükler, ölçekler ve bir OpenGL display list'e derler (hızlı çizim)."""
    tris, normals, colors = load_model(path)
    tris = fit_to_size(tris, (SCREEN_W_CM * MODEL_FILL, SCREEN_H_CM * MODEL_FILL,
                              ROOM_DEPTH_CM * MODEL_FILL))
    lst = glGenLists(1)
    glNewList(lst, GL_COMPILE)
    glBegin(GL_TRIANGLES)
    for i, (tri, n) in enumerate(zip(tris, normals)):
        if colors is not None:
            glColor3f(*colors[i])
        glNormal3f(*n)
        for vtx in tri:
            glVertex3f(*vtx)
    glEnd()
    glEndList()
    return lst


class Renderer:
    def __init__(self, size=(1280, 720), model_path=None):
        pygame.init()
        pygame.display.set_caption("Aşama 3 - Off-Axis Parallaks")
        self.size = size
        pygame.display.set_mode(size, DOUBLEBUF | OPENGL | RESIZABLE)
        glClearColor(0.02, 0.02, 0.06, 1.0)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glEnable(GL_LINE_SMOOTH)
        glLineWidth(1.8)
        self.angle = 0.0
        self.model_list = _compile_model(model_path) if model_path else None
        self.show_room = self.model_list is None   # model varken tel kafes çizgiler gizli
        self.model_scale = 1.0                     # çalışırken PageUp/PageDown ile değişir

    def resize(self, size):
        self.size = size
        glViewport(0, 0, *size)

    def _draw_model(self):
        # Işık dünya sabit (kamera ötelemesinden sonra, model dönüşümünden önce)
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_LIGHTING)
        glEnable(GL_LIGHT0)
        glEnable(GL_NORMALIZE)
        glEnable(GL_COLOR_MATERIAL)
        glLightfv(GL_LIGHT0, GL_POSITION, (0.3, 0.6, 1.0, 0.0))
        glLightfv(GL_LIGHT0, GL_DIFFUSE, (1.0, 1.0, 1.0, 1.0))
        glLightfv(GL_LIGHT0, GL_AMBIENT, (0.25, 0.25, 0.3, 1.0))
        glColor4f(0.8, 0.8, 0.85, 1.0)   # renk yoksa varsayılan gri
        glCallList(self.model_list)
        glDisable(GL_LIGHTING)
        glDisable(GL_DEPTH_TEST)

    def draw(self, eye, dt=0.0):
        """eye: ekran merkezine göre (ex, ey, ez) cm."""
        self.angle += 25.0 * dt
        l, r, b, t, n, f = off_axis_frustum(eye, NEAR, FAR)

        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        glFrustum(l, r, b, t, n, f)

        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()
        glTranslatef(-eye[0], -eye[1], -eye[2])   # kamera ötelenir, DÖNMEZ

        if self.show_room:
            _draw_room()
        glPushMatrix()
        glTranslatef(0.0, 0.0, -ROOM_DEPTH_CM * 0.5)
        if self.model_list is not None:
            glScalef(self.model_scale, self.model_scale, self.model_scale)
            self._draw_model()
        else:
            glRotatef(self.angle, 0.3, 1.0, 0.0)
            _draw_cube((0.0, 0.0, 0.0), 8.0)
        glPopMatrix()

        pygame.display.flip()
