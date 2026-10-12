"""Point cloud loading, cleaning, plane extraction and scale estimation."""
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import open3d as o3d


@dataclass
class Plane:
    """A RANSAC plane ax + by + cz + d = 0 and the points that support it."""
    model: np.ndarray        # (4,) unit normal + offset
    points: np.ndarray       # (N, 3) inlier points

    @property
    def normal(self) -> np.ndarray:
        return self.model[:3]

    def tilt_deg(self, up=(0.0, 0.0, 1.0)) -> float:
        """Angle between the plane normal and the up axis (0 = horizontal)."""
        c = abs(float(np.dot(self.normal, np.asarray(up, dtype=float))))
        return float(np.degrees(np.arccos(np.clip(c, 0.0, 1.0))))


def load_cloud(path) -> o3d.geometry.PointCloud:
    pcd = o3d.io.read_point_cloud(str(Path(path)))
    if pcd.is_empty():
        raise RuntimeError(f"Point cloud is empty or unreadable: {path}")
    return pcd


def clean_cloud(pcd, voxel_size=0.03, nb_neighbors=20, std_ratio=2.0):
    """Downsample then drop statistical outliers.

    voxel_size is in the cloud's own units, which are arbitrary until scaled.
    """
    down = pcd.voxel_down_sample(voxel_size=voxel_size)
    cleaned, _ = down.remove_statistical_outlier(
        nb_neighbors=nb_neighbors, std_ratio=std_ratio
    )
    return cleaned


def extract_planes(pcd, distance_threshold=0.05, min_points=200,
                   max_planes=10, ransac_n=3, num_iterations=1000):
    """Iteratively peel the largest planes off the cloud with RANSAC."""
    planes = []
    rest = pcd
    while len(planes) < max_planes and len(rest.points) >= min_points:
        model, inliers = rest.segment_plane(
            distance_threshold=distance_threshold,
            ransac_n=ransac_n,
            num_iterations=num_iterations,
        )
        if len(inliers) < min_points:
            break
        model = np.asarray(model, dtype=float)
        model = model / np.linalg.norm(model[:3])
        pts = np.asarray(rest.select_by_index(inliers).points)
        planes.append(Plane(model=model, points=pts))
        rest = rest.select_by_index(inliers, invert=True)
    return planes


def split_clusters(points, eps, min_points=20):
    """Split one RANSAC plane's points into spatially connected patches, largest first."""
    pts = np.asarray(points, float)
    if len(pts) == 0:
        return []
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts))
    labels = np.asarray(pcd.cluster_dbscan(eps=eps, min_points=min_points))
    groups = [pts[labels == k] for k in range(labels.max() + 1)] if labels.size else []
    return sorted(groups, key=len, reverse=True)


def scale_factor(measured_length, p1, p2) -> float:
    """Real units per cloud unit, from one measured distance between two cloud points."""
    model_len = float(np.linalg.norm(np.asarray(p2, float) - np.asarray(p1, float)))
    if model_len <= 0:
        raise ValueError("Reference points coincide; cannot set scale")
    if measured_length <= 0:
        raise ValueError("Measured length must be positive")
    return measured_length / model_len
