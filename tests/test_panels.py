import numpy as np
import open3d as o3d
import pytest
from shapely.geometry import box

from rooftop import panels as P
from rooftop.cloud import split_clusters
from rooftop.geometry import PlaneFrame

FLAT = PlaneFrame(np.zeros(3), np.array([1.0, 0, 0]), np.array([0, 1.0, 0]), np.array([0, 0, 1.0]))


def test_polygon_ring_drops_closing_vertex():
    assert P.polygon_ring(box(0, 0, 1, 2)).shape == (4, 2)


def test_panel_corners_lifted_by_standoff():
    c = P.panel_corners_3d(FLAT, [box(0, 0, 1, 2)], standoff=0.1)[0]
    assert c.shape == (4, 3)
    assert np.allclose(c[:, 2], 0.1)
    assert np.ptp(c[:, 0]) == pytest.approx(1) and np.ptp(c[:, 1]) == pytest.approx(2)


def test_panel_corners_follow_tilted_frame():
    n = np.array([0, -np.sin(0.5), np.cos(0.5)])
    v = np.cross(n, [1, 0, 0])
    f = PlaneFrame(np.zeros(3), np.array([1.0, 0, 0]), v, n)
    c = P.panel_corners_3d(f, [box(0, 0, 1, 1)], standoff=0)[0]
    assert np.allclose(c @ n, 0)


def test_sample_ring_and_quad():
    ring = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0.0]])
    r = P.sample_ring(ring, 0.1)
    assert len(r) == 40
    q = P.sample_quad(ring, 0.25)
    assert q.min(0) == pytest.approx([0, 0, 0]) and q.max(0) == pytest.approx([1, 1, 0])


def test_write_overlay_colours(tmp_path):
    pts = np.random.rand(100, 3)
    path = P.write_overlay(tmp_path / "o.ply", pts, None,
                           P.panel_corners_3d(FLAT, [box(0, 0, 1, 1)]),
                           FLAT.to_3d(P.polygon_ring(box(0, 0, 2, 2))))
    c = np.asarray(o3d.io.read_point_cloud(str(path)).colors)
    assert len(c) > 100
    assert (np.abs(c - [0.1, 0.3, 0.95]).sum(1) < 0.02).any()  # blue panel points
    assert (np.abs(c - [1.0, 0.1, 0.1]).sum(1) < 0.02).any()   # red outline points


def test_export_step_solids_and_volumes(tmp_path):
    import cadquery as cq
    panels = [box(1, 1, 2, 3), box(3, 1, 4, 3)]
    path = P.export_step(tmp_path / "r.step", FLAT, box(0, 0, 10, 6),
                         P.panel_corners_3d(FLAT, panels, standoff=0.1),
                         obstacles=box(6, 2, 7, 3), panel_thickness=0.04, roof_thickness=0.05)
    shape = cq.importers.importStep(str(path))
    solids = shape.solids().vals()
    assert len(solids) == 4  # roof + keep-out + 2 panels
    vols = sorted(s.Volume() for s in solids)
    mm3 = 1e9
    assert vols[0] == pytest.approx(1 * 2 * 0.04 * mm3, rel=1e-6)
    assert vols[-1] == pytest.approx(60 * 0.05 * mm3, rel=1e-6)
    bb = shape.val().BoundingBox()
    assert bb.zmax == pytest.approx(500)  # keep-out 0.5 m tall, panels at 0.1 + 0.04


def test_plot_plan_writes_png(tmp_path):
    p = P.plot_plan(tmp_path / "p.png", np.random.rand(50, 2), box(0, 0, 1, 1),
                    box(0.1, 0.1, 0.9, 0.9), box(0.4, 0.4, 0.5, 0.5), [box(0.1, 0.1, 0.3, 0.3)])
    assert p.read_bytes()[:4] == b"\x89PNG"


def test_split_clusters_separates_patches():
    rng = np.random.default_rng(0)
    a = np.c_[rng.uniform(0, 2, (300, 2)), np.zeros(300)]
    b = np.c_[rng.uniform(10, 11, (100, 2)), np.zeros(100)]
    groups = split_clusters(np.vstack([b, a]), eps=0.3, min_points=5)
    assert [len(g) for g in groups[:2]] == [300, 100]
    assert split_clusters(np.zeros((0, 3)), eps=1) == []
