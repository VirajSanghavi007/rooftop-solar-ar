import numpy as np
import open3d as o3d
import pytest

from rooftop.cloud import Plane, clean_cloud, extract_planes, load_cloud, scale_factor


def _pcd(pts):
    return o3d.geometry.PointCloud(o3d.utility.Vector3dVector(np.asarray(pts, float)))


def _rooftop_cloud(rng):
    """Flat floor z=0 (10x6) plus a vertical wall x=0, with small noise."""
    floor = np.c_[rng.uniform(0, 10, 3000), rng.uniform(0, 6, 3000), rng.normal(0, 0.005, 3000)]
    wall = np.c_[rng.normal(0, 0.005, 1500), rng.uniform(0, 6, 1500), rng.uniform(0.2, 1.0, 1500)]
    return _pcd(np.vstack([floor, wall]))


def test_load_cloud_rejects_empty(tmp_path):
    p = tmp_path / "empty.ply"
    o3d.io.write_point_cloud(str(p), _pcd(np.zeros((0, 3))))
    with pytest.raises(RuntimeError):
        load_cloud(p)


def test_load_cloud_roundtrip(tmp_path):
    p = tmp_path / "c.ply"
    o3d.io.write_point_cloud(str(p), _pcd(np.random.rand(50, 3)))
    assert len(load_cloud(p).points) == 50


def test_clean_cloud_removes_far_outlier():
    rng = np.random.default_rng(1)
    pts = np.vstack([rng.uniform(0, 1, (2000, 3)), [[50, 50, 50]]])
    out = np.asarray(clean_cloud(_pcd(pts), voxel_size=0.02).points)
    assert out.max() < 2
    assert len(out) > 500


def test_extract_planes_finds_floor_and_wall():
    o3d.utility.random.seed(0)
    planes = extract_planes(_rooftop_cloud(np.random.default_rng(2)),
                            distance_threshold=0.02, min_points=500)
    tilts = sorted(p.tilt_deg() for p in planes[:2])
    assert len(planes) >= 2
    assert tilts[0] < 2          # floor horizontal
    assert abs(tilts[1] - 90) < 2  # wall vertical
    assert len(planes[0].points) >= 2500


def test_extract_planes_stops_below_min_points():
    assert extract_planes(_pcd(np.random.rand(50, 3)), min_points=200) == []


def test_plane_tilt():
    assert Plane(np.array([0, 0, 1, 0.0]), np.zeros((0, 3))).tilt_deg() == pytest.approx(0)
    assert Plane(np.array([1, 0, 0, 0.0]), np.zeros((0, 3))).tilt_deg() == pytest.approx(90)


def test_scale_factor():
    assert scale_factor(12.0, [0, 0, 0], [3, 4, 0]) == pytest.approx(2.4)


@pytest.mark.parametrize("m,p2", [(5, [0, 0, 0]), (0, [1, 0, 0]), (-1, [1, 0, 0])])
def test_scale_factor_rejects_bad_input(m, p2):
    with pytest.raises(ValueError):
        scale_factor(m, [0, 0, 0], p2)
