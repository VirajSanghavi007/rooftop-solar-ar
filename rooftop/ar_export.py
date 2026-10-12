"""Export the roof + panel design as glTF binary (.glb) for AR viewers.

glTF is metres, Y-up. The model origin is put at the roof-plane origin so the
AR app can anchor it to a real point on the roof (corner tap, marker, geo pose).
"""
from pathlib import Path

import numpy as np
import shapely

# z-up (our levelled frame) -> y-up (glTF)
Z_TO_Y = np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]], float)

PANEL_RGB = (0.08, 0.15, 0.45)
ROOF_RGB = (0.55, 0.55, 0.55)


def box_from_quad(corners, thickness):
    """Vertices (8,3) and triangles (12,3) of a thin slab on top of a planar quad."""
    c = np.asarray(corners, float)
    n = np.cross(c[1] - c[0], c[3] - c[0])
    n = n / np.linalg.norm(n)
    if n[2] < 0:
        n = -n
    v = np.vstack([c - n * thickness, c])
    t = np.array([[0, 2, 1], [0, 3, 2], [4, 5, 6], [4, 6, 7],
                  [0, 1, 5], [0, 5, 4], [1, 2, 6], [1, 6, 5],
                  [2, 3, 7], [2, 7, 6], [3, 0, 4], [3, 4, 7]])
    return v, t


def polygon_mesh(poly, frame, h=0.0):
    """Triangulate a (possibly concave, holed) roof polygon onto its plane."""
    tris = shapely.constrained_delaunay_triangles(poly)
    verts, faces = [], []
    for tri in tris.geoms:
        ring = np.asarray(tri.exterior.coords)[:3]
        base = len(verts)
        verts.extend(frame.to_3d(ring, h))
        faces.append([base, base + 1, base + 2])
    return np.asarray(verts), np.asarray(faces)


def build_scene(frame, outline, panel_corners, panel_thickness=0.035):
    """List of (vertices, triangles, rgb) parts in the levelled z-up frame."""
    parts = []
    if outline is not None and not outline.is_empty:
        v, t = polygon_mesh(outline, frame, h=-0.005)
        parts.append((v, t, ROOF_RGB))
    for c in panel_corners:
        v, t = box_from_quad(c, panel_thickness)
        parts.append((v, t, PANEL_RGB))
    return parts


def _pad4(b, fill=b"\x00"):
    return b + fill * (-len(b) % 4)


def glb_bytes(parts, metallic=0.1, roughness=0.6):
    """Minimal glTF 2.0 binary: one primitive + material per colour, uint32 indices.

    parts: [(vertices (N,3) y-up metres, triangles (M,3), rgb)]
    Open3D's own .glb writer produces out-of-range buffer views, hence this.
    """
    import json
    import struct

    by_rgb = {}
    for v, t, rgb in parts:
        acc = by_rgb.setdefault(tuple(rgb), ([], [], 0))
        vs, ts, n = acc
        vs.append(np.asarray(v, np.float32))
        ts.append(np.asarray(t, np.uint32) + n)
        by_rgb[tuple(rgb)] = (vs, ts, n + len(v))

    blob, views, accessors, prims, materials = b"", [], [], [], []
    for i, (rgb, (vs, ts, _)) in enumerate(by_rgb.items()):
        pos, idx = np.vstack(vs), np.vstack(ts).ravel()
        for data, target in ((pos.tobytes(), 34962), (idx.tobytes(), 34963)):
            views.append({"buffer": 0, "byteOffset": len(blob), "byteLength": len(data), "target": target})
            blob = _pad4(blob + data)
        accessors.append({"bufferView": len(views) - 2, "componentType": 5126, "count": len(pos),
                          "type": "VEC3", "min": pos.min(0).tolist(), "max": pos.max(0).tolist()})
        accessors.append({"bufferView": len(views) - 1, "componentType": 5125,
                          "count": int(idx.size), "type": "SCALAR"})
        materials.append({"pbrMetallicRoughness": {"baseColorFactor": [*rgb, 1.0],
                                                   "metallicFactor": metallic,
                                                   "roughnessFactor": roughness},
                          "doubleSided": True})
        prims.append({"attributes": {"POSITION": 2 * i}, "indices": 2 * i + 1,
                      "material": i, "mode": 4})
    gltf = {"asset": {"version": "2.0", "generator": "rooftop.ar_export"},
            "scene": 0, "scenes": [{"nodes": [0]}], "nodes": [{"mesh": 0, "name": "RoofSolar"}],
            "meshes": [{"primitives": prims}], "materials": materials,
            "accessors": accessors, "bufferViews": views, "buffers": [{"byteLength": len(blob)}]}
    js = _pad4(json.dumps(gltf, separators=(",", ":")).encode(), b" ")
    total = 12 + 8 + len(js) + 8 + len(blob)
    return (struct.pack("<4sII", b"glTF", 2, total)
            + struct.pack("<I4s", len(js), b"JSON") + js
            + struct.pack("<I4s", len(blob), b"BIN\x00") + blob)


def write_glb(path, frame, outline, panel_corners, panel_thickness=0.035):
    """Single .glb, y-up metres, origin at the roof-plane origin."""
    parts = [((v - frame.origin) @ Z_TO_Y.T, t, rgb)
             for v, t, rgb in build_scene(frame, outline, panel_corners, panel_thickness)]
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(glb_bytes(parts))
    return path
