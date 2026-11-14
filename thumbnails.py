"""
Menü önizlemeleri. Her model bir kez ekran dışında (flip'siz) çizilip küçük PNG olarak
`.thumbs/` klasörüne kaydedilir; dosya değişince (mtime/boyut) önbellek anahtarı değişir ve yenilenir.
"""

import os

import cv2
import numpy as np

THUMB_W, THUMB_H = 320, 180
THUMBS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".thumbs")
WIREFRAME_KEY = "wireframe-v1"   # modelsiz (tel kafes oda) önizlemesi için sabit anahtar


def thumb_path(model_path, thumbs_dir=THUMBS_DIR):
    """
    :param model_path: model dosyası yolu ya da None (tel kafes oda)
    :return: önbellek PNG yolu. Dosyanın mtime'ı ve boyutu anahtarda olduğundan değişen model yeniden üretilir.
    """
    if model_path is None:
        return os.path.join(thumbs_dir, WIREFRAME_KEY + ".png")
    st = os.stat(model_path)
    stem = os.path.splitext(os.path.basename(model_path))[0]
    return os.path.join(thumbs_dir, f"{stem}-{int(st.st_mtime)}-{st.st_size}.png")


def has_thumbnail(model_path, thumbs_dir=THUMBS_DIR):
    return os.path.exists(thumb_path(model_path, thumbs_dir))


def _prune_stale(model_path, keep, thumbs_dir):
    """Aynı modelin eski (mtime/boyutu farklı) önizlemelerini siler."""
    if model_path is None or not os.path.isdir(thumbs_dir):
        return
    stem = os.path.splitext(os.path.basename(model_path))[0]
    for name in os.listdir(thumbs_dir):
        path = os.path.join(thumbs_dir, name)
        # "<stem>-<mtime>-<size>.png": stem'den sonra iki sayısal parça gelmeli (başka modelin adıyla karışmasın)
        if path != keep and name.startswith(stem + "-"):
            rest = name[len(stem) + 1:-4].split("-")
            if len(rest) == 2 and all(p.isdigit() for p in rest):
                os.remove(path)


def ensure_thumbnail(renderer, model_path, thumbs_dir=THUMBS_DIR):
    """
    Önizleme yoksa üretir. Modeli yükleyip ekran dışında çizer; kayıtlı görünümünü (.view.json) kullanır.
    NOT: renderer.load_model çağırır — o an yüklü modelin yerini alır (çağıran gerekirse geri yüklemeli).
    :return: PNG yolu
    """
    path = thumb_path(model_path, thumbs_dir)
    if os.path.exists(path):
        return path
    os.makedirs(thumbs_dir, exist_ok=True)
    renderer.load_model(model_path)
    renderer.draw((0.0, 0.0, 60.0), 0.0, present=False)
    frame = renderer.grab_frame()
    small = cv2.resize(frame, (THUMB_W, THUMB_H), interpolation=cv2.INTER_AREA)
    cv2.imwrite(path, cv2.cvtColor(small, cv2.COLOR_RGB2BGR))
    _prune_stale(model_path, path, thumbs_dir)
    return path


def load_thumbnail_rgb(path, width=160):
    """
    PNG'yi okuyup glDrawPixels'e uygun ham RGB baytlarına çevirir (alt satır önce).
    :return: (bayt, genişlik, yükseklik) | dosya okunamazsa None
    """
    img = cv2.imread(path)
    if img is None:
        return None
    h = int(round(img.shape[0] * width / img.shape[1]))
    img = cv2.resize(img, (width, h), interpolation=cv2.INTER_AREA)
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return np.ascontiguousarray(np.flipud(rgb)).tobytes(), width, h
