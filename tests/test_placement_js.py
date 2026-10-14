"""Unit tests for preview/placement.js (corner-tap placement math), run through node."""
import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
JS = (ROOT / "preview" / "placement.js").as_uri()


@pytest.fixture(scope="module")
def node_available():
    if not shutil.which("node"):
        pytest.skip("node not installed")


def run(expr):
    code = f'import * as P from "{JS}"; console.log(JSON.stringify({expr}));'
    out = subprocess.run(["node", "--input-type=module", "-e", code], capture_output=True, text=True, check=True)
    return json.loads(out.stdout)


def test_rotate_y_matches_three_convention(node_available):
    # three.js rotation.y = +90 deg maps +x to -z
    assert np.allclose(run("P.rotateY([1, 0, 0], Math.PI / 2)"), [0, 0, -1], atol=1e-12)


def test_true_size_maps_corners_exactly(node_available):
    a, b = [2, 0.1, 1], [12, 0.1, 1]           # 10 m model edge along +x
    A, B = [5, 0, 5], [5, 0, -5]               # tapped 10 m along -z
    r = run(f"(() => {{ const p = P.solvePlacement({a}, {b}, {A}, {B}); "
            f"return [p, P.applyPlacement(p, {a}), P.applyPlacement(p, {b}), P.applyPlacement(p, [7, 0.1, 3])] }})()")
    pl, pa, pb, pm = r
    assert pl["scale"] == 1 and pl["errorPct"] == pytest.approx(0)
    assert np.allclose(pa, A, atol=1e-9) and np.allclose(pb, B, atol=1e-9)
    # a point 2 m to the +z side of the edge (model) stays 2 m away, rigid motion
    assert np.linalg.norm(np.subtract(pm, pa)) == pytest.approx(np.hypot(5, 2))


def test_fit_scales_to_tapped_length_and_reports_error(node_available):
    pl, pb = run("(() => { const p = P.solvePlacement([0,0,0],[10,0,0],[1,0,1],[1,0,9], true); "
                 "return [p, P.applyPlacement(p, [10,0,0])] })()")
    assert pl["scale"] == pytest.approx(0.8) and pl["errorPct"] == pytest.approx(-20)
    assert np.allclose(pb, [1, 0, 9], atol=1e-9)
    pl = run("P.solvePlacement([0,0,0],[10,0,0],[0,0,0],[8,0,0], false)")
    assert pl["scale"] == 1 and pl["tappedLength"] == pytest.approx(8)


def test_degenerate_taps_rejected(node_available):
    with pytest.raises(subprocess.CalledProcessError):
        run("P.solvePlacement([0,0,0],[10,0,0],[1,0,1],[1,0,1])")
