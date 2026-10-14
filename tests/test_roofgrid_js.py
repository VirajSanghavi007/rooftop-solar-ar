"""Unit tests for preview/roofgrid.js (live 4-corner panel layout), run through node."""
import json
import math
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
JS = (ROOT / "preview" / "roofgrid.js").as_uri()


@pytest.fixture(scope="module")
def node_available():
    if not shutil.which("node"):
        pytest.skip("node not installed")


def run(expr):
    code = f'import * as G from "{JS}"; console.log(JSON.stringify({expr}));'
    out = subprocess.run(["node", "--input-type=module", "-e", code], capture_output=True, text=True)
    if out.returncode:
        raise RuntimeError(out.stderr)
    return json.loads(out.stdout)


def test_order_corners_any_tap_order_keeps_a_then_b(node_available):
    # A, B, then the far corners tapped in "wrong" order
    r = run("G.orderCorners([[0,0],[10,0],[0,6],[10,6]])")
    assert r[:2] == [[0, 0], [10, 0]] and sorted(map(tuple, r[2:])) == [(0, 6), (10, 6)]
    assert r[2] == [10, 6]                                 # walks around the perimeter
    with pytest.raises(RuntimeError, match="neighbouring"):
        run("G.orderCorners([[0,0],[10,6],[10,0],[0,6]])")  # B opposite A


def test_inset_and_convexity(node_available):
    r = run("G.insetConvex([[0,0],[10,0],[10,6],[0,6]], 0.5)")
    assert sorted(map(tuple, r)) == [(0.5, 0.5), (0.5, 5.5), (9.5, 0.5), (9.5, 5.5)]
    assert run("G.insetConvex([[0,0],[2,0],[2,2],[0,2]], 1.5)") is None
    assert run("G.isConvex([[0,0],[10,0],[2,2],[0,6]])") is False


def test_panels_keep_real_size_count_scales_with_roof(node_available):
    small = run("G.layoutRoof([[0,0],[5,0],[5,4],[0,4]], {setback: 0.5})")
    big = run("G.layoutRoof([[0,0],[20,0],[20,12],[0,12]], {setback: 0.5})")
    for r in (small, big):
        for p in r["panels"]:
            assert sorted([p["width"], p["depth"]]) == [1.134, 1.722]   # never scaled
    # 4 x 3 usable: 2 landscape across x 2 rows (1.722*2, 1.134*2) fits, not 3 rows
    assert len(small["panels"]) == 4
    # 19 x 11 usable: landscape 10 x 9 = 90, portrait 16 x 6 = 96 -> portrait wins
    assert len(big["panels"]) == 96 and big["orientation"] == "portrait"


def test_panels_inside_tilted_roof_any_rotation(node_available):
    ang = math.radians(33)
    rot = lambda x, z: [round(x * math.cos(ang) - z * math.sin(ang), 9), round(x * math.sin(ang) + z * math.cos(ang), 9)]
    pts = [rot(0, 0), rot(12, 0), rot(12, 8), rot(0, 8)]
    r = run(f"(() => {{ const L = G.layoutRoof({pts}); return [L.panels.length, "
            f"L.panels.every(p => G.rectFits([[p.s,p.t],[p.s+p.width,p.t],[p.s+p.width,p.t+p.foot],[p.s,p.t+p.foot]]"
            f".map(L.toXZ), L.roof, 0.5 - 1e-6))] }})()")
    n, inside = r
    assert inside and n == run("G.layoutRoof([[0,0],[12,0],[12,8],[0,8]]).panels.length")


def test_tilt_spaces_rows_for_winter_shadow(node_available):
    alt = run("G.winterSunAltitude(20)")
    assert alt == pytest.approx(46.56)
    gap = run(f"G.rowGap(1.134, 15, {alt})")
    assert gap == pytest.approx(1.134 * math.sin(math.radians(15)) / math.tan(math.radians(alt)))
    flat = run("G.layoutRoof([[0,0],[12,0],[12,10],[0,10]])")
    tilt = run(f"G.layoutRoof([[0,0],[12,0],[12,10],[0,10]], {{tiltDeg: 15, sunAltDeg: {alt}}})")
    assert tilt["rowPitch"] > tilt["depth"] * math.cos(math.radians(15))
    assert len(tilt["panels"]) <= len(flat["panels"])


RECT = "(p) => [[p.s,p.t],[p.s+p.width,p.t],[p.s+p.width,p.t+p.foot],[p.s,p.t+p.foot]]"


def test_l_shaped_roof_concave_corner(node_available):
    # 12 x 10 L: the 6 x 5 top-right notch is not roof
    L_ROOF = "[[0,0],[12,0],[12,5],[6,5],[6,10],[0,10]]"
    r = run(f"(() => {{ const R = {RECT}; const L = G.layoutRoof({L_ROOF}); return [L.panels.length, "
            f"L.panels.every(p => G.rectFits(R(p).map(L.toXZ), L.roof, 0.5 - 1e-6)), "
            f"L.panels.some(p => R(p).map(L.toXZ).some(q => q[0] > 6.01 && q[1] > 4.99))] }})()")
    n, all_fit, any_in_notch = r
    full = run("G.layoutRoof([[0,0],[12,0],[12,10],[0,10]]).panels.length")
    assert all_fit and not any_in_notch
    assert 0.55 * full < n < 0.85 * full          # L is 75 % of the box, minus extra setback


def test_obstacle_blocks_panels_with_clearance(node_available):
    roof = "[[0,0],[12,0],[12,8],[0,8]]"
    tank = "[[5,3],[7,3],[7,5],[5,5]]"
    free = run(f"G.layoutRoof({roof}).panels.length")
    r = run(f"(() => {{ const R = {RECT}; const L = G.layoutRoof({roof}, {{obstacles: [{tank}], clearance: 0.3}}); "
            f"return [L.panels.length, Math.min(...L.panels.map(p => G.polyDist(R(p).map(L.toXZ), {tank})))] }})()")
    n, closest = r
    assert 0 < n < free and closest >= 0.3 - 1e-9
    # 2-tap obstacle (box diagonal) behaves like the same box tapped as 4 corners
    assert run(f"G.layoutRoof({roof}, {{obstacles: [[[5,3],[7,5]]]}}).panels.length") == n


def test_box_from_diagonal_follows_roof_axes(node_available):
    s = math.sqrt(0.5)
    box = run(f"G.boxFromDiagonal([0,0],[2,0],[{s},{s}],[{-s},{s}])")
    assert len(box) == 4 and box[0] == [0, 0]
    assert abs(math.dist(box[0], box[1]) - math.sqrt(2)) < 1e-9   # side along u


def test_crossing_edges_rejected(node_available):
    with pytest.raises(RuntimeError, match="cross"):
        run("G.layoutRoof([[0,0],[10,0],[0,8],[10,8],[5,12]])")     # bow-tie 5-gon
    assert run("G.isSimple([[0,0],[10,0],[10,8],[0,8]])") is True
    assert run("G.pointInPolygon([[0,0],[12,0],[12,5],[6,5],[6,10],[0,10]], [9,8])") is False
