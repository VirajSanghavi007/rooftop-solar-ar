"""Load the COLMAP dense cloud, clean it, report planes, save cleaned_cloud.ply.

Usage: python scripts/inspect_cloud.py [site] [--voxel 0.03] [--no-view]
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import open3d as o3d

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from rooftop.cloud import clean_cloud, extract_planes, load_cloud  # noqa: E402


def find_fused(site_dir: Path) -> Path:
    """Dense cloud: OpenMVS scene_dense.ply preferred, else COLMAP fused.ply."""
    mvs = site_dir / "mvs" / "scene_dense.ply"
    if mvs.is_file():
        return mvs
    hits = sorted((site_dir / "colmap").rglob("fused.ply"))
    if not hits:
        raise FileNotFoundError(f"No mvs/scene_dense.ply or colmap/**/fused.ply under {site_dir}")
    return hits[0]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("site", nargs="?", default="rooftop_01")
    ap.add_argument("--voxel", type=float, default=0.03)
    ap.add_argument("--no-view", action="store_true")
    a = ap.parse_args(argv)

    src = find_fused(ROOT / "data" / a.site)
    out = ROOT / "outputs" / a.site
    out.mkdir(parents=True, exist_ok=True)

    pcd = load_cloud(src)
    print(f"Loaded {len(pcd.points)} points from {src}")
    print("Extent (cloud units, NOT metres yet):", pcd.get_axis_aligned_bounding_box().get_extent())

    pcd = clean_cloud(pcd, voxel_size=a.voxel)
    print(f"Cleaned: {len(pcd.points)} points")
    o3d.io.write_point_cloud(str(out / "cleaned_cloud.ply"), pcd)

    planes = extract_planes(pcd, distance_threshold=a.voxel * 1.5)
    colours = np.random.default_rng(0).random((len(planes), 3))
    preview = []
    for i, (pl, c) in enumerate(zip(planes, colours)):
        print(f"Plane {i}: {len(pl.points)} pts, tilt {pl.tilt_deg():.1f} deg, model {np.round(pl.model, 3)}")
        g = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pl.points))
        g.paint_uniform_color(c)
        preview.append(g)
    if preview:
        merged = preview[0]
        for g in preview[1:]:
            merged += g
        o3d.io.write_point_cloud(str(out / "roof_preview.ply"), merged)

    if not a.no_view:
        o3d.visualization.draw_geometries(preview or [pcd])


if __name__ == "__main__":
    main()
