"""
3B model yükleyici (OpenGL'den bağımsız).
  - .obj         : basit kendi ayrıştırıcımız (yalnızca geometri, gri)
  - .glb / .gltf : trimesh ile; her parça kendi dokusu (texture) ve UV'leriyle gelir

Çıktı: parça listesi. Her parça bir sözlük:
  vertices (V,3) float32, normals (V,3) float32, faces (F,3) uint32,
  uv (V,2) float32 | None, image PIL.Image | None, color (4,) float32 0..1,
  vcolors (V,4) float32 0..1 | None
"""

import os
import numpy as np


def load_model(path):
    ext = os.path.splitext(path)[1].lower()
    if ext == ".obj":
        return [_part_from_triangles(load_obj(path))]
    if ext in (".glb", ".gltf"):
        return load_gltf(path)
    raise ValueError(f"Desteklenmeyen format '{ext}'. .obj, .glb veya .gltf kullanın.")


def model_bounds(parts):
    """:return: (lo, hi) tüm parçaları kapsayan kutu."""
    allv = np.concatenate([p["vertices"] for p in parts])
    return allv.min(axis=0), allv.max(axis=0)


# --------------------------------------------------------------------------- glTF

def load_gltf(path):
    import trimesh
    scene = trimesh.load(path)
    if isinstance(scene, trimesh.Trimesh):
        instances = [(scene, np.eye(4))]
    else:
        # Sahne grafiğini kendimiz dolaşıyoruz: scene.dump() her parça için dokuları
        # kopyalıyor ve büyük sahnelerde GB'larca bellek harcıyor.
        instances = []
        for node in scene.graph.nodes_geometry:
            transform, geom_name = scene.graph[node]
            instances.append((scene.geometry[geom_name], transform))
    parts = [p for p in (_part_from_mesh(m, T) for m, T in instances) if p is not None]
    if not parts:
        raise ValueError(f"{path}: çizilebilir geometri bulunamadı.")
    return parts


def _part_from_mesh(m, transform):
    import trimesh
    if not isinstance(m, trimesh.Trimesh) or len(m.faces) == 0:
        return None
    T = np.asarray(transform, dtype=np.float64)
    verts = m.vertices @ T[:3, :3].T + T[:3, 3]
    normals = m.vertex_normals @ np.linalg.inv(T[:3, :3])   # normal dönüşümü: (M^-1)^T
    faces = np.asarray(m.faces)
    if np.linalg.det(T[:3, :3]) < 0:                        # aynalama: üçgen yönünü düzelt
        faces = faces[:, ::-1]
    part = {
        "vertices": np.ascontiguousarray(verts, dtype=np.float32),
        "normals": np.ascontiguousarray(normals, dtype=np.float32),
        "faces": np.ascontiguousarray(faces, dtype=np.uint32),
        "uv": None, "image": None, "vcolors": None,
        "color": np.array([0.8, 0.8, 0.85, 1.0], dtype=np.float32),
    }
    vis = m.visual
    if vis.kind == "texture":
        mat = vis.material
        factor = getattr(mat, "baseColorFactor", None)
        if factor is not None:
            part["color"] = np.asarray(factor, dtype=np.float32) / 255.0
        elif getattr(mat, "diffuse", None) is not None:
            part["color"] = np.asarray(mat.diffuse, dtype=np.float32) / 255.0
        image = getattr(mat, "baseColorTexture", None) or getattr(mat, "image", None)
        if image is not None and vis.uv is not None and len(vis.uv) == len(m.vertices):
            part["image"] = image
            part["uv"] = np.ascontiguousarray(vis.uv, dtype=np.float32)
    elif vis.kind in ("vertex", "face"):
        part["vcolors"] = np.ascontiguousarray(vis.vertex_colors, dtype=np.float32) / 255.0
    return part


# --------------------------------------------------------------------------- OBJ

def load_obj(path):
    """:return: (N, 3, 3) float32 üçgen köşe konumları"""
    verts, tris = [], []
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
    return v[np.asarray(tris, dtype=np.int64)]


def _part_from_triangles(tris):
    """Üçgen soup -> düz gölgeli parça (her üçgen kendi köşeleri ve yüz normaliyle)."""
    n = np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0])
    length = np.linalg.norm(n, axis=1, keepdims=True)
    n = n / np.where(length == 0, 1.0, length)
    return {
        "vertices": np.ascontiguousarray(tris.reshape(-1, 3), dtype=np.float32),
        "normals": np.ascontiguousarray(np.repeat(n, 3, axis=0), dtype=np.float32),
        "faces": np.arange(len(tris) * 3, dtype=np.uint32).reshape(-1, 3),
        "uv": None, "image": None, "vcolors": None,
        "color": np.array([0.8, 0.8, 0.85, 1.0], dtype=np.float32),
    }
