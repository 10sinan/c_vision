"""
Off-Axis Projeksiyon Matematiği (Aşama 3) - OpenGL'den bağımsız, test edilebilir.

Koordinat sistemi (birim: cm):
  - Monitör düzlemi z = 0, merkezi orijin. Ekran köşeleri (±W/2, ±H/2, 0).
  - Oda ekranın ARKASINA doğru uzanır (z < 0). Kullanıcı z > 0 tarafındadır.
  - X: sağ +, Y: yukarı +.
"""

import math

# 15.6" 16:9 monitör (fiziksel ölçü, cm)
SCREEN_W_CM = 34.5
SCREEN_H_CM = 19.4
# Webcam ekranın üst orta kenarında varsayılır: ekran merkezinin ~ H/2 + 1 cm üstü
CAMERA_OFFSET_Y_CM = SCREEN_H_CM / 2.0 + 1.0


def head_to_eye_cm(x_n, y_n, z_cm, frame_w, frame_h, focal_px,
                   cam_offset_y_cm=CAMERA_OFFSET_Y_CM):
    """
    Normalize kafa konumu (x_n, y_n in [-1,1], z_cm) -> ekran merkezine göre
    göz konumu (ex, ey, ez) cm.

    Pinhole modeli: X_cm = x_px_merkezden * Z / f   =>   X_cm = x_n * (W/2) * Z / f
    Kamera ekranın üstünde olduğu için ey'ye kamera ofseti eklenir.
    """
    ex = x_n * (frame_w / 2.0) * z_cm / focal_px
    ey = y_n * (frame_h / 2.0) * z_cm / focal_px + cam_offset_y_cm
    return ex, ey, z_cm


def amplify_eye(eye, gain, center_y_cm=0.0):
    """
    Baş hareketini güçlendirir: gözün yatay/dikey konumu, nötr noktaya (x=0, y=center_y_cm) göre
    gain ile çarpılır; uzaklık (z) değişmez. gain=1 fiziksel olarak doğrudur. Otururken baş yalnızca
    birkaç cm oynar ve camın arkasındaki her şey aynı yöne kayar; yakın-uzak farkı (paralaks) ancak
    daha büyük hareketle belirginleşir, bu yüzden etki için hareket büyütülür.
    """
    ex, ey, ez = eye
    return ex * gain, center_y_cm + (ey - center_y_cm) * gain, ez


def off_axis_frustum(eye, near, far,
                     screen_w=SCREEN_W_CM, screen_h=SCREEN_H_CM):
    """
    Göz konumuna göre asimetrik frustum sınırları (glFrustum parametreleri).
    Benzer üçgenler: ekran kenarının gözden görünüşü near düzlemine ölçeklenir.

      left   = (-W/2 - ex) * near / ez      right = ( W/2 - ex) * near / ez
      bottom = (-H/2 - ey) * near / ez      top   = ( H/2 - ey) * near / ez

    Kamera ekran düzlemine dik kalır (dönmez); sadece frustum kayar.
    :return: (left, right, bottom, top, near, far)
    """
    ex, ey, ez = eye
    ez = max(ez, 1.0)  # sıfıra bölmeyi engelle
    k = near / ez
    return ((-screen_w / 2.0 - ex) * k,
            (screen_w / 2.0 - ex) * k,
            (-screen_h / 2.0 - ey) * k,
            (screen_h / 2.0 - ey) * k,
            near, far)
