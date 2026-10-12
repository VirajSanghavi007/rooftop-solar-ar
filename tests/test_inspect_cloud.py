import sys
from pathlib import Path

import numpy as np
import open3d as o3d
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import inspect_cloud  # noqa: E402


def _touch(p):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("")
    return p


def test_find_fused_prefers_openmvs(tmp_path):
    _touch(tmp_path / "colmap/dense/0/fused.ply")
    mvs = _touch(tmp_path / "mvs/scene_dense.ply")
    assert inspect_cloud.find_fused(tmp_path) == mvs


def test_find_fused_falls_back_to_colmap(tmp_path):
    f = _touch(tmp_path / "colmap/dense/0/fused.ply")
    assert inspect_cloud.find_fused(tmp_path) == f


def test_find_fused_missing(tmp_path):
    with pytest.raises(FileNotFoundError):
        inspect_cloud.find_fused(tmp_path)


def test_main_smoke(tmp_path, monkeypatch):
    """End to end on a tiny synthetic site: cleaned + preview clouds written."""
    rng = np.random.default_rng(0)
    floor = np.c_[rng.uniform(0, 5, 3000), rng.uniform(0, 5, 3000), rng.normal(0, 0.003, 3000)]
    site = tmp_path / "data/site"
    (site / "mvs").mkdir(parents=True)
    o3d.io.write_point_cloud(str(site / "mvs/scene_dense.ply"),
                             o3d.geometry.PointCloud(o3d.utility.Vector3dVector(floor)))
    monkeypatch.setattr(inspect_cloud, "ROOT", tmp_path)
    inspect_cloud.main(["site", "--voxel", "0.05", "--no-view"])
    out = tmp_path / "outputs/site"
    assert len(o3d.io.read_point_cloud(str(out / "cleaned_cloud.ply")).points) > 100
    assert (out / "roof_preview.ply").is_file()
