"""Measure the panel design's real accuracy against ETH3D laser scans.

Usage: python scripts/validate_eth3d.py eth3d/terrace [design_panels options...]

Reports, all in metres in the levelled frame:
  cloud    : our dense points vs scan (accuracy) and scan vs ours on the roof (completeness)
  plane    : roof tilt / height vs a least-squares plane through the scan
  outline  : area error, IoU and edge distance vs an outline traced from the scan
  panels   : scan surface height under every panel corner, corners outside the true roof
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import shapely

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
import design_panels  # noqa: E402
from rooftop import layout as L  # noqa: E402
from rooftop import validate as V  # noqa: E402
from rooftop.panels import polygon_ring  # noqa: E402


def validate(site, gt_voxel=0.02, plane_band=0.10, **design_kw):
    import open3d as o3d

    summary, obj = design_panels.run(site, return_objects=True, **design_kw)
    R, frame, outline = obj["R"], obj["frame"], obj["outline"]
    voxel = design_kw.get("voxel", 0.05)
    print("[validate] loading laser scans")
    gt = V.load_ground_truth(ROOT / "data" / site / "dslr_scan_eval", voxel=gt_voxel)
    gt.rotate(R, center=np.zeros(3))
    ours = o3d.io.read_point_cloud(str(design_panels.find_fused(ROOT / "data" / site)))
    ours = ours.voxel_down_sample(voxel)
    ours.rotate(R, center=np.zeros(3))

    # cloud accuracy (ours -> scan) restricted to the roof footprint +1 m
    region = outline.buffer(1.0)
    o_uv, o_h = frame.to_2d(np.asarray(ours.points))
    keep = shapely.contains_xy(region, o_uv[:, 0], o_uv[:, 1]) & (np.abs(o_h) < 3)
    acc = V.distance_stats(np.asarray(ours.select_by_index(np.flatnonzero(keep))
                                      .compute_point_cloud_distance(gt)))

    # scan points on the roof plane, in our roof frame
    g_uv, g_h = frame.to_2d(np.asarray(gt.points))
    on = (np.abs(g_h) < plane_band) & shapely.contains_xy(region, g_uv[:, 0], g_uv[:, 1])
    roof_gt = np.asarray(gt.points)[on]
    comp = V.distance_stats(np.asarray(
        o3d.geometry.PointCloud(o3d.utility.Vector3dVector(roof_gt)).compute_point_cloud_distance(ours)))

    n_gt, c_gt = V.fit_plane_lsq(roof_gt)
    plane = {
        "tilt_ours_deg": frame.tilt_deg(),
        "tilt_truth_deg": float(np.degrees(np.arccos(abs(n_gt[2])))),
        "normal_angle_err_deg": float(np.degrees(np.arccos(min(1.0, abs(n_gt @ frame.n))))),
        "height_offset_m": float((c_gt - frame.origin) @ frame.n),
    }

    truth_outline = L.roof_outline(g_uv[on], ratio=design_kw.get("hull_ratio", 0.15))
    out_m = V.outline_metrics(outline, truth_outline)

    corner_uv = np.vstack([polygon_ring(p) for p in obj["layout"].panels]) if obj["layout"].count else np.zeros((0, 2))
    h_err = V.corner_height_errors(corner_uv, g_uv[on], g_h[on]) if len(corner_uv) else np.array([])
    inside = V.corners_inside(corner_uv, truth_outline) if len(corner_uv) else np.array([], bool)
    finite = h_err[np.isfinite(h_err)]
    panels = {
        "corners": int(len(corner_uv)),
        "corners_outside_true_roof": int((~inside).sum()),
        "corner_height_abs_median_m": float(np.median(np.abs(finite))) if finite.size else None,
        "corner_height_abs_max_m": float(np.abs(finite).max()) if finite.size else None,
        "corners_without_scan_support": int((~np.isfinite(h_err)).sum()),
    }

    report = {"site": site, "design": {k: summary[k] for k in ("panels", "kwp", "outline_area_m2", "usable_area_m2")},
              "cloud_accuracy": acc, "roof_completeness": comp, "plane": plane,
              "outline": out_m, "panels": panels}
    path = ROOT / "outputs" / site / "validation.json"
    path.write_text(json.dumps(report, indent=2))
    return report


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("site", nargs="?", default="eth3d/terrace")
    ap.add_argument("--voxel", type=float, default=0.05)
    a = ap.parse_args(argv)
    r = validate(a.site, voxel=a.voxel)
    print(json.dumps(r, indent=2))


if __name__ == "__main__":
    main()
