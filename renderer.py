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
    GL_LINE_SMOOTH,
)

from off_axis import off_axis_frustum, SCREEN_W_CM, SCREEN_H_CM

ROOM_DEPTH_CM = 40.0
GRID_STEPS = 8          # duvarlardaki derinlik çizgisi sayısı
NEAR, FAR = 1.0, 300.0


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


class Renderer:
    def __init__(self, size=(1280, 720)):
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

    def resize(self, size):
        self.size = size
        glViewport(0, 0, *size)

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

        _draw_room()
        glPushMatrix()
        glTranslatef(0.0, 0.0, -ROOM_DEPTH_CM * 0.45)
        glRotatef(self.angle, 0.3, 1.0, 0.0)
        _draw_cube((0.0, 0.0, 0.0), 8.0)
        glPopMatrix()

        pygame.display.flip()
