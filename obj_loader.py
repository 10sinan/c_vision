"""
3B model yükleyici.
  - .obj  : basit kendi ayrıştırıcımız (yalnızca geometri, renk yok)
  - .glb / .gltf : trimesh ile (geometri + malzeme/doku renkleri, yüz başına renk)
Çokgenler üçgene bölünür; yüz normalleri (düz gölgelendirme) burada hesaplanır.
"""

import os
import numpy as np


def load_model(path):
    """
    :return: (triangles (N,3,3), normals (N,3), colors (N,3) 0..1 | None)
    """
    ext = os.path.splitext(path)[1].lower()
    if ext == ".obj":
        t, n = load_obj(path)
        return t, n, None
    if ext in (".glb", ".gltf"):
        return load_gltf(path)
    raise ValueError(f"Desteklenmeyen format '{ext}'. .obj, .glb veya .gltf kullanın.")


def load_gltf(path):
    import trimesh
    mesh = trimesh.load(path, force="mesh")   # sahneyi tek mesh'e birleştirir (dönüşümler uygulanır)
    t = np.asarray(mesh.triangles, dtype=np.float32)
    n = np.asarray(mesh.face_normals, dtype=np.float32)
    colors = None
    try:
        # Doku/malzeme renklerini köşe renklerine çevirir, yüz başına ortalama alır
        vis = mesh.visual
        if vis.kind == "texture":
            vis = vis.to_color()   # dokudan renk örnekler
        vc = np.asarray(vis.vertex_colors, dtype=np.float32)[:, :3] / 255.0
        colors = vc[mesh.faces].mean(axis=1).astype(np.float32)
    except Exception:
        pass  # renk çıkarılamazsa gri çizilir
    return t, n, colors


def load_obj(path):
    """
    :return: (triangles, normals)
      triangles: (N, 3, 3) float32 köşe konumları
      normals:   (N, 3)    float32 birim yüz normalleri
    """
    verts = []
    tris = []
    with open(path, "r", errors="ignore") as fh:
        for line in fh:
            parts = line.split()
            if not parts:
                continue
            if parts[0] == "v":
                verts.append([float(p) for p in parts[1:4]])
            elif parts[0] == "f":
                idx = []
                for tok in parts[1:]:
                    i = int(tok.split("/")[0])
                    idx.append(i - 1 if i > 0 else len(verts) + i)  # negatif indeks desteği
                for k in range(1, len(idx) - 1):                    # fan üçgenleme
                    tris.append((idx[0], idx[k], idx[k + 1]))
    if not verts or not tris:
        raise ValueError(f"{path}: geçerli geometri bulunamadı (v/f satırları yok).")

    v = np.asarray(verts, dtype=np.float32)
    t = v[np.asarray(tris, dtype=np.int64)]
    n = np.cross(t[:, 1] - t[:, 0], t[:, 2] - t[:, 0])
    length = np.linalg.norm(n, axis=1, keepdims=True)
    n = n / np.where(length == 0, 1.0, length)
    return t.astype(np.float32), n.astype(np.float32)


def fit_to_size(triangles, box_cm):
    """Modeli orijine ortalar ve (genişlik, yükseklik, derinlik) kutusuna sığacak şekilde ölçekler."""
    flat = triangles.reshape(-1, 3)
    lo, hi = flat.min(axis=0), flat.max(axis=0)
    extent = np.maximum(hi - lo, 1e-9)
    scale = float(np.min(np.asarray(box_cm, dtype=np.float32) / extent))
    return (triangles - (lo + hi) / 2.0) * scale
