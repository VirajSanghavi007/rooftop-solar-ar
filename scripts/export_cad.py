"""Build rooftop CAD from measurements.json and export STEP + DXF for AutoCAD.

Usage: python scripts/export_cad.py [site]
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from rooftop.cad import RooftopSpec, export_dxf, export_step, rooftop_boxes  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("site", nargs="?", default="rooftop_01")
    a = ap.parse_args(argv)

    spec = RooftopSpec.from_json(ROOT / "data" / a.site / "measurements.json")
    boxes = rooftop_boxes(spec)
    out = ROOT / "outputs" / a.site
    print("STEP:", export_step(boxes, out / "rooftop.step"))
    print("DXF: ", export_dxf(boxes, out / "rooftop.dxf"))


if __name__ == "__main__":
    main()
