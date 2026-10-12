"""Levelling, plane frames and 2D <-> 3D projection."""
from dataclasses import dataclass
from pathlib import Path

import numpy as np

Z = np.array([0.0, 0.0, 1.0])


def unit(v):
    v = np.asarray(v, dtype=float)
    n = np.linalg.norm(v)
    if n == 0:
        raise ValueError("Zero-length vector")
    return v / n


def rotation_between(a, b) -> np.ndarray:
    """3x3 rotation taking direction a onto direction b (Rodrigues)."""
    a, b = unit(a), unit(b)
    v = np.cross(a, b)
    c = float(np.dot(a, b))
    if np.isclose(c, 1.0):
        return np.eye(3)
    if np.isclose(c, -1.0):
        # 180 deg about any axis perpendicular to a
        axis = unit(np.cross(a, [1, 0, 0] if abs(a[0]) < 0.9 else [0, 1, 0]))
        return 2 * np.outer(axis, axis) - np.eye(3)
    k = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + k + k @ k * (1 / (1 + c))


def estimate_up(planes, reference_points) -> np.ndarray:
    """Up = normal of the largest plane, flipped so reference points sit above it.

    reference_points: camera centres (best) or the whole cloud. Photos are taken
    from above the ground/roof, so most references lie on the positive side.
    """
    if not planes:
        raise ValueError("No planes to level against")
    big = max(planes, key=lambda p: len(p.points))
    n, d = unit(big.model[:3]), big.model[3] / np.linalg.norm(big.model[:3])
    side = np.asarray(reference_points, float) @ n + d
    off = side[np.abs(side) > 0.02 * np.ptp(side)]  # ignore points lying on the plane
    if off.size == 0:
        off = side
    return n if np.median(off) >= 0 else -n


def level_transform(up) -> np.ndarray:
    return rotation_between(up, Z)


def read_camera_centres(images_txt) -> np.ndarray:
    """Camera centres C = -R^T t from a COLMAP images.txt."""
    centres = []
    lines = [l for l in Path(images_txt).read_text().splitlines() if l and not l.startswith("#")]
    for line in lines[::2]:  # every other line is the 2D point list
        f = line.split()
        qw, qx, qy, qz, tx, ty, tz = map(float, f[1:8])
        r = np.array([
            [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
            [2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
            [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)],
        ])
        centres.append(-r.T @ np.array([tx, ty, tz]))
    return np.array(centres)


@dataclass
class PlaneFrame:
    """Orthonormal frame on a plane: origin o, in-plane axes u, v, normal n."""
    origin: np.ndarray
    u: np.ndarray
    v: np.ndarray
    n: np.ndarray

    @classmethod
    def from_points(cls, points, normal, up=Z):
        n = unit(normal)
        if np.dot(n, up) < 0:
            n = -n
        # u: world X projected into the plane (stable for near-horizontal roofs)
        ref = np.array([1.0, 0, 0]) if abs(n[0]) < 0.9 else np.array([0, 1.0, 0])
        u = unit(ref - np.dot(ref, n) * n)
        v = np.cross(n, u)
        return cls(np.asarray(points, float).mean(axis=0), u, v, n)

    def to_2d(self, points):
        """(N,3) -> (N,2) in-plane coords and (N,) height above the plane."""
        d = np.asarray(points, float) - self.origin
        return np.c_[d @ self.u, d @ self.v], d @ self.n

    def to_3d(self, uv, h=0.0):
        uv = np.asarray(uv, float)
        h = np.broadcast_to(np.asarray(h, float), (len(uv),))
        return self.origin + uv[:, :1] * self.u + uv[:, 1:2] * self.v + h[:, None] * self.n

    def tilt_deg(self, up=Z) -> float:
        return float(np.degrees(np.arccos(np.clip(abs(np.dot(self.n, up)), 0, 1))))
