"""Compare a reconstruction + panel design against a laser-scan ground truth."""
import re
from pathlib import Path

import numpy as np
import shapely
from shapely.geometry import Point


def parse_mlp(path):
    """MeshLab project -> {scan filename: 4x4 matrix mapping scan into the image frame}."""
    text = Path(path).read_text()
    out = {}
    for name, mat in re.findall(r'filename="([^"]+)".*?<MLMatrix44>(.*?)</MLMatrix44>', text, re.S):
        out[name] = np.array(mat.split(), float).reshape(4, 4)
    if not out:
        raise ValueError(f"No scans found in {path}")
    return out


def load_ground_truth(scan_dir, voxel=0.02):
    """All scans in scan_dir merged into the image frame, downsampled."""
    import open3d as o3d

    scan_dir = Path(scan_dir)
    gt = o3d.geometry.PointCloud()
    for name, m in parse_mlp(scan_dir / "scan_alignment.mlp").items():
        p = o3d.io.read_point_cloud(str(scan_dir / name)).voxel_down_sample(voxel)
        gt += p.transform(m)
    return gt


def distance_stats(d, thresholds=(0.02, 0.05, 0.10)):
    """Median / p90 and the share of distances under each threshold (metres)."""
    d = np.asarray(d, float)
    if d.size == 0:
        raise ValueError("No distances")
    out = {"median_m": float(np.median(d)), "p90_m": float(np.percentile(d, 90))}
    for t in thresholds:
        out[f"within_{int(round(t * 100))}cm"] = float((d <= t).mean())
    return out


def fit_plane_lsq(points):
    """Least-squares plane through points: (unit normal with +z, centroid)."""
    pts = np.asarray(points, float)
    c = pts.mean(axis=0)
    n = np.linalg.svd(pts - c, full_matrices=False)[2][-1]
    return (n if n[2] >= 0 else -n), c


def outline_metrics(ours, truth):
    """Area error, IoU and boundary (edge) distances between two outlines."""
    inter = ours.intersection(truth).area
    union = ours.union(truth).area
    b_ours, b_truth = ours.exterior, truth.exterior
    pts = [b_ours.interpolate(f, normalized=True) for f in np.linspace(0, 1, 400, endpoint=False)]
    edge = np.array([b_truth.distance(p) for p in pts])
    return {
        "area_ours_m2": ours.area, "area_truth_m2": truth.area,
        "area_error_pct": 100 * (ours.area - truth.area) / truth.area,
        "iou": inter / union if union else 0.0,
        "edge_median_m": float(np.median(edge)), "edge_p90_m": float(np.percentile(edge, 90)),
        "hausdorff_m": float(ours.hausdorff_distance(truth)),
    }


def corner_height_errors(corners_uv, truth_uv, truth_h, radius=0.15):
    """Truth roof height (median of scan points within radius) at each panel corner.

    Returns errors in metres (truth height relative to our roof plane, i.e. how far
    the real surface sits from where we placed the panel base) and NaN where the
    scan has no points.
    """
    from scipy.spatial import cKDTree

    tree = cKDTree(np.asarray(truth_uv))
    out = []
    for q in np.asarray(corners_uv):
        idx = tree.query_ball_point(q, radius)
        out.append(float(np.median(np.asarray(truth_h)[idx])) if idx else np.nan)
    return np.array(out)


def corners_inside(corners_uv, outline):
    return np.array([outline.covers(Point(*c)) for c in np.asarray(corners_uv)])
