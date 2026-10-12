import numpy as np
import pytest

from rooftop.cloud import Plane
from rooftop.geometry import (PlaneFrame, Z, estimate_up, level_transform,
                              read_camera_centres, rotation_between)


def _random_rotation(seed):
    q = np.random.default_rng(seed).normal(size=4)
    w, x, y, z = q / np.linalg.norm(q)
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


@pytest.mark.parametrize("a", [[1, 2, 3], [0, 0, 1], [0, 0, -1], [1, 0, 0], [-0.3, 0.1, -5]])
def test_rotation_between_maps_a_to_b_and_is_proper(a):
    R = rotation_between(a, Z)
    assert np.allclose(R @ (np.array(a) / np.linalg.norm(a)), Z, atol=1e-9)
    assert np.allclose(R @ R.T, np.eye(3), atol=1e-9)
    assert np.linalg.det(R) == pytest.approx(1)


def test_rotation_between_rejects_zero():
    with pytest.raises(ValueError):
        rotation_between([0, 0, 0], Z)


def _ground_plane(normal, d=0.0, n=100):
    return Plane(np.r_[normal, d], np.zeros((n, 3)))


def test_estimate_up_flips_towards_cameras():
    plane = _ground_plane([0, 0, -1])  # normal pointing down
    cams = np.array([[0, 0, 5.0], [1, 1, 4.0]])
    assert np.allclose(estimate_up([plane], cams), [0, 0, 1])


def test_estimate_up_with_cloud_ignores_on_plane_points():
    plane = _ground_plane([0, 0, -1])
    rng = np.random.default_rng(0)
    floor = np.c_[rng.uniform(0, 10, (1000, 2)), np.zeros(1000)]
    above = np.c_[rng.uniform(0, 1, (50, 2)), rng.uniform(0.5, 2, 50)]
    assert np.allclose(estimate_up([plane], np.vstack([floor, above])), [0, 0, 1])


def test_estimate_up_picks_largest_plane():
    small = Plane(np.array([1.0, 0, 0, 0]), np.zeros((10, 3)))
    big = _ground_plane([0, 0, 1], n=500)
    assert np.allclose(estimate_up([small, big], [[0, 0, 3.0]]), [0, 0, 1])


def test_estimate_up_requires_planes():
    with pytest.raises(ValueError):
        estimate_up([], [[0, 0, 1]])


def test_level_transform_undoes_random_tilt():
    R0 = _random_rotation(3)
    up_tilted = R0 @ Z
    assert np.allclose(level_transform(up_tilted) @ up_tilted, Z)


def test_read_camera_centres(tmp_path):
    # identity rotation, t = (1,2,3)  ->  C = -t
    # 90 deg about z (qw=qz=sqrt(.5)), t=(1,0,0) -> C = -R^T t = (0, 1, 0)
    s = np.sqrt(0.5)
    p = tmp_path / "images.txt"
    p.write_text("# header\n"
                 "1 1 0 0 0 1 2 3 1 a.jpg\n10 20 -1\n"
                 f"2 {s} 0 0 {s} 1 0 0 1 b.jpg\n\n")
    c = read_camera_centres(p)
    assert np.allclose(c[0], [-1, -2, -3])
    assert np.allclose(c[1], [0, 1, 0], atol=1e-9)


def test_plane_frame_roundtrip_and_orthonormal():
    rng = np.random.default_rng(1)
    pts = rng.normal(size=(50, 3))
    f = PlaneFrame.from_points(pts, [0.2, -0.1, 1.0])
    M = np.vstack([f.u, f.v, f.n])
    assert np.allclose(M @ M.T, np.eye(3), atol=1e-9)
    uv, h = f.to_2d(pts)
    assert np.allclose(f.to_3d(uv, h), pts)


def test_plane_frame_normal_points_up_and_tilt():
    f = PlaneFrame.from_points(np.zeros((3, 3)), [0, 0, -1])
    assert np.allclose(f.n, [0, 0, 1])
    assert f.tilt_deg() == pytest.approx(0)
    assert PlaneFrame.from_points(np.zeros((3, 3)), [1, 0, 1]).tilt_deg() == pytest.approx(45)
