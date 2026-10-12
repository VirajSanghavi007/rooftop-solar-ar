import json
from itertools import combinations

import pytest

from rooftop.cad import RooftopSpec, export_dxf, export_step, rooftop_boxes

SPEC = RooftopSpec(length=12000, width=8000, slab_thickness=150,
                   parapet_height=900, parapet_thickness=200)


def _overlap(a, b):
    return all(a.origin[i] < b.origin[i] + b.size[i] and b.origin[i] < a.origin[i] + a.size[i]
               for i in range(3))


def test_boxes_names_and_count():
    names = [b.name for b in rooftop_boxes(SPEC)]
    assert names == ["RoofSlab", "FrontParapet", "BackParapet", "LeftParapet", "RightParapet"]


def test_boxes_do_not_overlap():
    for a, b in combinations(rooftop_boxes(SPEC), 2):
        assert not _overlap(a, b), (a.name, b.name)


def test_boxes_fit_footprint_and_height():
    for b in rooftop_boxes(SPEC):
        assert b.origin[0] >= 0 and b.origin[0] + b.size[0] <= SPEC.length
        assert b.origin[1] >= 0 and b.origin[1] + b.size[1] <= SPEC.width
        assert b.origin[2] + b.size[2] <= SPEC.slab_thickness + SPEC.parapet_height


def test_total_volume():
    s = SPEC
    expected = (s.length * s.width * s.slab_thickness
                + (s.length * s.width - (s.length - 2 * s.parapet_thickness)
                   * (s.width - 2 * s.parapet_thickness)) * s.parapet_height)
    vol = sum(b.size[0] * b.size[1] * b.size[2] for b in rooftop_boxes(s))
    assert vol == pytest.approx(expected)


@pytest.mark.parametrize("kw", [{"length": 0}, {"width": -1}, {"parapet_thickness": 5000}])
def test_spec_validation(kw):
    base = dict(length=12000, width=8000)
    base.update(kw)
    with pytest.raises(ValueError):
        RooftopSpec(**base)


def test_spec_from_json_ignores_extra_keys(tmp_path):
    p = tmp_path / "m.json"
    p.write_text(json.dumps({"_note": "x", "diagonal": None, "length": 5000, "width": 4000}))
    s = RooftopSpec.from_json(p)
    assert (s.length, s.width, s.parapet_height) == (5000, 4000, 900)


def test_export_step_volume(tmp_path):
    import cadquery as cq
    path = export_step(rooftop_boxes(SPEC), tmp_path / "r.step")
    shape = cq.importers.importStep(str(path)).val()
    expected = sum(b.size[0] * b.size[1] * b.size[2] for b in rooftop_boxes(SPEC))
    assert shape.Volume() == pytest.approx(expected, rel=1e-6)


def test_export_dxf_layers_and_bounds(tmp_path):
    import ezdxf
    path = export_dxf(rooftop_boxes(SPEC), tmp_path / "r.dxf")
    doc = ezdxf.readfile(path)
    meshes = list(doc.modelspace().query("POLYLINE"))
    assert {m.dxf.layer for m in meshes} == {b.name for b in rooftop_boxes(SPEC)}
    zs = [v.dxf.location.z for m in meshes for v in m.vertices if v.is_poly_face_mesh_vertex]
    assert max(zs) == pytest.approx(SPEC.slab_thickness + SPEC.parapet_height)
