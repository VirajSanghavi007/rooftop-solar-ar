"""Smoke test on a synthetic tilted rooftop + integration test on real ETH3D terrace."""
import json
import sys
from pathlib import Path

import numpy as np
import open3d as o3d
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import design_panels  # noqa: E402


def _synthetic_roof(rng):
    """12 x 8 m flat roof, 1 m parapets, a 1.5 m chimney, then randomly rotated."""
    def area(x0, x1, y0, y1, z, n):
        return np.c_[rng.uniform(x0, x1, n), rng.uniform(y0, y1, n), np.full(n, z)]

    roof = area(0, 12, 0, 8, 0, 40000)
    roof = roof[~((roof[:, 0] > 5) & (roof[:, 0] < 6) & (roof[:, 1] > 3) & (roof[:, 1] < 4))]
    walls = [np.c_[rng.uniform(0, 12, 3000), np.full(3000, y), rng.uniform(0, 1, 3000)] for y in (0, 8)]
    walls += [np.c_[np.full(2000, x), rng.uniform(0, 8, 2000), rng.uniform(0, 1, 2000)] for x in (0, 12)]
    chim = np.vstack([area(5, 6, 3, 4, z, 150) for z in np.linspace(0.1, 1.5, 10)])
    pts = np.vstack([roof, *walls, chim])
    pts = pts + rng.normal(0, 0.005, pts.shape)
    ang = np.radians([25, -10, 40])
    rx = np.array([[1, 0, 0], [0, np.cos(ang[0]), -np.sin(ang[0])], [0, np.sin(ang[0]), np.cos(ang[0])]])
    ry = np.array([[np.cos(ang[1]), 0, np.sin(ang[1])], [0, 1, 0], [-np.sin(ang[1]), 0, np.cos(ang[1])]])
    rz = np.array([[np.cos(ang[2]), -np.sin(ang[2]), 0], [np.sin(ang[2]), np.cos(ang[2]), 0], [0, 0, 1]])
    return pts @ (rz @ ry @ rx).T


def test_design_panels_smoke_synthetic(tmp_path, monkeypatch):
    site = tmp_path / "data" / "synth"
    (site / "mvs").mkdir(parents=True)
    o3d.io.write_point_cloud(str(site / "mvs" / "scene_dense.ply"), o3d.geometry.PointCloud(
        o3d.utility.Vector3dVector(_synthetic_roof(np.random.default_rng(0)))))
    monkeypatch.setattr(design_panels, "ROOT", tmp_path)

    s = design_panels.run("synth", voxel=0.1, setback=0.5, max_tilt=10)

    assert s["roof_tilt_deg"] < 1.0                       # levelled despite 25/10/40 deg rotation
    assert s["outline_area_m2"] == pytest.approx(96, rel=0.05)
    # usable: (12-1)x(8-1)=77 minus chimney (1 m + 0.3 clearance each side ~ 2.6 m2)
    assert 70 < s["usable_area_m2"] < 76
    assert 25 <= s["panels"] <= 38                        # 1.134 x 1.722 m panels
    for f in s["files"].values():
        assert Path(f).stat().st_size > 0
    assert json.loads((tmp_path / "outputs/synth/panels_summary.json").read_text())["panels"] == s["panels"]
    anchor = json.loads((tmp_path / "outputs/synth/panels_anchor.json").read_text())
    assert anchor["anchor_length_m"] == pytest.approx(12, rel=0.05)   # long edge of the 12 x 8 roof


TERRACE = ROOT / "data" / "eth3d" / "terrace" / "mvs" / "scene_dense.ply"


@pytest.fixture
def terrace_available():
    if not TERRACE.is_file():
        pytest.skip("ETH3D terrace dense cloud not built (see README)")


@pytest.mark.integration
def test_design_panels_eth3d_terrace(terrace_available):
    s = design_panels.run("eth3d/terrace", voxel=0.05)
    assert s["roof_tilt_deg"] < 5
    assert s["outline_area_m2"] > 5
    assert s["panels"] >= 1
