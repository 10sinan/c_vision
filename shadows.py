"""
Gölge haritası (shadow mapping) — tek yönlü ana ışık için.

Sahne ve ışık dünyaya sabittir, kafa hareketi gölgeyi değiştirmez; bu yüzden gölge haritası yalnızca
model/görünüm (ölçek, kaydırma, döndürme) değişince yeniden hesaplanır (bake). Her karede yalnızca
gölge dokusu okunur.

Sürücü desteklemezse (GLSL 1.20 derlenmez, FBO tamamlanmaz) `available` False olur ve renderer
sabit-fonksiyonlu hatta (gölgesiz) döner.

Matematik (light_matrices) OpenGL'siz test edilir; GL kısmı Shadows sınıfındadır.
"""

import os

import numpy as np

SHADOW_SIZE = 2048

# --------------------------------------------------------------------------- saf matematik


def _normalize(v):
    v = np.asarray(v, dtype=np.float64)
    return v / np.linalg.norm(v)


def look_at(eye, target, up):
    """Sağ-el sistemi görünüm matrisi (4x4, satır-ana; sütun vektörlerle çarpılır)."""
    f = _normalize(np.asarray(target, dtype=np.float64) - np.asarray(eye, dtype=np.float64))
    if abs(float(f @ _normalize(up))) > 0.99:           # ışık yönü yukarı vektörüyle neredeyse paralel
        up = (1.0, 0.0, 0.0)
    s = _normalize(np.cross(f, up))
    u = np.cross(s, f)
    m = np.eye(4)
    m[0, :3], m[1, :3], m[2, :3] = s, u, -f
    m[:3, 3] = -m[:3, :3] @ np.asarray(eye, dtype=np.float64)
    return m


def ortho(left, right, bottom, top, near, far):
    """OpenGL glOrtho eşdeğeri (4x4, satır-ana)."""
    m = np.eye(4)
    m[0, 0] = 2.0 / (right - left)
    m[1, 1] = 2.0 / (top - bottom)
    m[2, 2] = -2.0 / (far - near)
    m[0, 3] = -(right + left) / (right - left)
    m[1, 3] = -(top + bottom) / (top - bottom)
    m[2, 3] = -(far + near) / (far - near)
    return m


def light_matrices(center, radius, light_dir):
    """
    Merkezi `center`, yarıçapı `radius` olan küreyi tamamen kapsayan ışık görünüm/projeksiyon matrisleri.
    :param light_dir: ışığa doğru yön (yüzeyden ışığa bakan vektör)
    :return: (V, P) — 4x4 satır-ana float64; clip = P @ V @ [dünya, 1]
    """
    center = np.asarray(center, dtype=np.float64)
    d = _normalize(light_dir)
    eye = center + d * radius * 2.0
    V = look_at(eye, center, (0.0, 0.0, 1.0))
    P = ortho(-radius, radius, -radius, radius, radius * 0.5, radius * 3.5)
    return V, P


# --------------------------------------------------------------------------- GL kısmı

VERTEX_SHADER = """
#version 120
uniform vec3 uEye;                 // göz konumu (dünya): kamera yalnızca ötelendiği için dünya = göz uzayı + uEye
uniform mat4 uLightVP;
varying vec3 vNormal;
varying vec4 vShadowPos;
varying float vFogDepth;
varying vec4 vColor;
void main() {
    vec4 eyePos = gl_ModelViewMatrix * gl_Vertex;
    vec3 world = eyePos.xyz + uEye;
    vShadowPos = uLightVP * vec4(world, 1.0);
    vNormal = gl_NormalMatrix * gl_Normal;
    vFogDepth = abs(eyePos.z);
    vColor = gl_Color;
    gl_TexCoord[0] = gl_MultiTexCoord0;
    gl_ClipVertex = eyePos;                        // kullanıcı kesme düzlemi (ekran camının önünü kes) shader'da bunu ister
    gl_Position = gl_ProjectionMatrix * eyePos;
}
"""

FRAGMENT_SHADER = """
#version 120
uniform sampler2D uTex;
uniform sampler2D uShadow;
uniform float uTexel;              // 1 / gölge haritası boyutu
uniform float uShadowStrength;     // 0..1 (1: tam karanlık gölge)
uniform float uFogOn;
varying vec3 vNormal;
varying vec4 vShadowPos;
varying float vFogDepth;
varying vec4 vColor;

float shadowFactor(float nl) {
    vec3 p = vShadowPos.xyz / vShadowPos.w * 0.5 + 0.5;
    if (p.x < 0.0 || p.x > 1.0 || p.y < 0.0 || p.y > 1.0 || p.z > 1.0) return 1.0;
    float bias = max(0.0025 * (1.0 - nl), 0.0007);
    float lit = 0.0;
    for (int i = -1; i <= 1; i++)
        for (int j = -1; j <= 1; j++) {
            float stored = texture2D(uShadow, p.xy + vec2(i, j) * uTexel).r;
            lit += (p.z - bias <= stored) ? 1.0 : 0.0;
        }
    return mix(1.0, lit / 9.0, uShadowStrength);
}

void main() {
    vec4 tex = texture2D(uTex, gl_TexCoord[0].st);
    vec4 base = tex * vColor;
    vec3 n = normalize(vNormal);
    if (!gl_FrontFacing) n = -n;                       // iki yüzlü aydınlatma (yaprak/çimen düzlemleri)
    vec3 l0 = normalize(gl_LightSource[0].position.xyz);
    vec3 l1 = normalize(gl_LightSource[1].position.xyz);
    float nl0 = max(dot(n, l0), 0.0);
    float nl1 = max(dot(n, l1), 0.0);
    float sh = nl0 > 0.0 ? shadowFactor(nl0) : 1.0;
    vec3 light = gl_LightModel.ambient.rgb
               + gl_LightSource[0].diffuse.rgb * nl0 * sh
               + gl_LightSource[1].diffuse.rgb * nl1;
    vec3 color = base.rgb * light;
    if (uFogOn > 0.5) {
        float f = clamp((gl_Fog.end - vFogDepth) * gl_Fog.scale, 0.0, 1.0);
        color = mix(gl_Fog.color.rgb, color, f);
    }
    gl_FragColor = vec4(color, base.a);
}
"""


class Shadows:
    """Gölge haritası kaynakları ve geçişleri. GL bağlamı hazır olduktan sonra oluşturulmalı."""

    def __init__(self, size=SHADOW_SIZE):
        self.size = size
        self.available = False
        self.key = None            # son bake'in anahtarı (değişince yeniden hesaplanır)
        self.light_vp = np.eye(4)
        self.reason = ""
        if os.environ.get("PARALLAX_NO_SHADOWS") == "1":
            self.reason = "PARALLAX_NO_SHADOWS=1"
            return
        try:
            self._init_gl()
            self.available = True
        except Exception as exc:       # sürücü/GLSL/FBO sorunu -> gölgesiz devam
            self.reason = f"{type(exc).__name__}: {exc}"
            print("[GÖLGE] Kullanılamıyor, gölgesiz devam:", self.reason)

    def _init_gl(self):
        from OpenGL.GL import (glGenTextures, glBindTexture, glTexImage2D, glTexParameteri, glTexParameterfv,
                               GL_TEXTURE_2D, GL_DEPTH_COMPONENT24, GL_DEPTH_COMPONENT, GL_FLOAT,
                               GL_TEXTURE_MIN_FILTER, GL_TEXTURE_MAG_FILTER, GL_NEAREST, GL_TEXTURE_WRAP_S,
                               GL_TEXTURE_WRAP_T, GL_CLAMP_TO_BORDER, GL_TEXTURE_BORDER_COLOR,
                               GL_TEXTURE_COMPARE_MODE, GL_NONE, glGenFramebuffers, glBindFramebuffer,
                               glFramebufferTexture2D, glDrawBuffer, glReadBuffer, glCheckFramebufferStatus,
                               GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_FRAMEBUFFER_COMPLETE,
                               GL_VERTEX_SHADER, GL_FRAGMENT_SHADER)
        from OpenGL.GL import shaders

        self.depth_tex = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, self.depth_tex)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_DEPTH_COMPONENT24, self.size, self.size, 0,
                     GL_DEPTH_COMPONENT, GL_FLOAT, None)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_BORDER)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_BORDER)
        glTexParameterfv(GL_TEXTURE_2D, GL_TEXTURE_BORDER_COLOR, (1.0, 1.0, 1.0, 1.0))   # dışarısı: aydınlık
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_COMPARE_MODE, GL_NONE)                  # derinliği elle karşılaştırırız

        self.fbo = glGenFramebuffers(1)
        glBindFramebuffer(GL_FRAMEBUFFER, self.fbo)
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_TEXTURE_2D, self.depth_tex, 0)
        glDrawBuffer(GL_NONE)
        glReadBuffer(GL_NONE)
        status = glCheckFramebufferStatus(GL_FRAMEBUFFER)
        glBindFramebuffer(GL_FRAMEBUFFER, 0)
        if status != GL_FRAMEBUFFER_COMPLETE:
            raise RuntimeError(f"FBO tamamlanmadı (0x{int(status):x})")

        self.program = shaders.compileProgram(
            shaders.compileShader(VERTEX_SHADER, GL_VERTEX_SHADER),
            shaders.compileShader(FRAGMENT_SHADER, GL_FRAGMENT_SHADER))

    def needs_bake(self, key):
        return key != self.key

    def bake(self, key, center, radius, light_dir, draw_model, viewport):
        """
        Gölge haritasını ışık gözünden yeniden çizer.
        :param key: değişince yeniden hesaplamayı tetikleyen anahtar (model + görünüm)
        :param draw_model: modeli (dönüşümleriyle) çizen parametresiz fonksiyon
        :param viewport: (x, y, w, h) — bitince geri yüklenir
        """
        from OpenGL.GL import (glBindFramebuffer, glViewport, glClear, glColorMask, glEnable, glDisable,
                               glMatrixMode, glPushMatrix, glPopMatrix, glLoadMatrixf, glPolygonOffset,
                               glAlphaFunc, GL_FRAMEBUFFER, GL_DEPTH_BUFFER_BIT, GL_FALSE, GL_TRUE,
                               GL_DEPTH_TEST, GL_POLYGON_OFFSET_FILL, GL_ALPHA_TEST, GL_GREATER,
                               GL_PROJECTION, GL_MODELVIEW, GL_LIGHTING, GL_FOG, GL_CLIP_PLANE0)
        V, P = light_matrices(center, radius, light_dir)
        self.light_vp = P @ V
        glBindFramebuffer(GL_FRAMEBUFFER, self.fbo)
        glViewport(0, 0, self.size, self.size)
        glClear(GL_DEPTH_BUFFER_BIT)
        glColorMask(GL_FALSE, GL_FALSE, GL_FALSE, GL_FALSE)
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_POLYGON_OFFSET_FILL)
        glPolygonOffset(2.0, 4.0)
        glDisable(GL_LIGHTING)
        glDisable(GL_FOG)
        glDisable(GL_CLIP_PLANE0)
        glEnable(GL_ALPHA_TEST)                          # yaprakların saydam pikselleri gölge düşürmesin
        glAlphaFunc(GL_GREATER, 0.4)
        glMatrixMode(GL_PROJECTION)
        glPushMatrix()
        glLoadMatrixf(np.ascontiguousarray(P.T, dtype=np.float32))
        glMatrixMode(GL_MODELVIEW)
        glPushMatrix()
        glLoadMatrixf(np.ascontiguousarray(V.T, dtype=np.float32))
        draw_model()
        glPopMatrix()
        glMatrixMode(GL_PROJECTION)
        glPopMatrix()
        glMatrixMode(GL_MODELVIEW)
        glDisable(GL_POLYGON_OFFSET_FILL)
        glDisable(GL_ALPHA_TEST)
        glDisable(GL_DEPTH_TEST)
        glColorMask(GL_TRUE, GL_TRUE, GL_TRUE, GL_TRUE)
        glBindFramebuffer(GL_FRAMEBUFFER, 0)
        glViewport(*viewport)
        self.key = key

    def begin_main_pass(self, eye, fog_on, strength=0.7):
        """Modeli gölgeli shader ile çizmeye hazırlar (glUseProgram + uniform'lar + gölge dokusu)."""
        from OpenGL.GL import (glUseProgram, glGetUniformLocation, glUniform1i, glUniform1f, glUniform3f,
                               glUniformMatrix4fv, glActiveTexture, glBindTexture, GL_TEXTURE0, GL_TEXTURE1,
                               GL_TEXTURE_2D, GL_TRUE)
        glUseProgram(self.program)
        loc = lambda name: glGetUniformLocation(self.program, name)
        glUniform1i(loc("uTex"), 0)
        glUniform1i(loc("uShadow"), 1)
        glUniform1f(loc("uTexel"), 1.0 / self.size)
        glUniform1f(loc("uShadowStrength"), float(strength))
        glUniform1f(loc("uFogOn"), 1.0 if fog_on else 0.0)
        glUniform3f(loc("uEye"), float(eye[0]), float(eye[1]), float(eye[2]))
        glUniformMatrix4fv(loc("uLightVP"), 1, GL_TRUE, np.ascontiguousarray(self.light_vp, dtype=np.float32))
        glActiveTexture(GL_TEXTURE1)
        glBindTexture(GL_TEXTURE_2D, self.depth_tex)
        glActiveTexture(GL_TEXTURE0)

    def end_main_pass(self):
        from OpenGL.GL import glUseProgram
        glUseProgram(0)

    def release(self):
        if not self.available:
            return
        from OpenGL.GL import glDeleteTextures, glDeleteFramebuffers, glDeleteProgram
        glDeleteTextures([self.depth_tex])
        glDeleteFramebuffers(1, [self.fbo])
        glDeleteProgram(self.program)
        self.available = False
