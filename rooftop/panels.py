"""Lift 2D panel layouts into 3D: STEP solids, overlay point clouds and a plan PNG."""
from pathlib import Path

import numpy as np


def polygon_ring(poly):
    """Exterior ring of a shapely polygon without the closing duplicate."""
    return np.asarray(poly.exterior.coords)[:-1, :2]


def panel_corners_3d(frame, panels, standoff=0.1, facing=None, rise=0.0):
    """List of (4,3) arrays: panel top surfaces above the roof plane.

    Flat: every corner at standoff. Tilted: height grows linearly away from the
    facing direction, front edge at standoff, back edge at standoff + rise.
    """
    out = []
    for p in panels:
        ring = polygon_ring(p)
        h = np.full(len(ring), standoff, float)
        if facing is not None and rise > 0:
            s = ring @ np.asarray(facing, float)
            h = standoff + rise * (s.max() - s) / np.ptp(s)
        out.append(frame.to_3d(ring, h))
    return out


def sample_ring(ring3d, step):
    """Points along a closed 3D polyline every `step` units."""
    out = []
    for a, b in zip(ring3d, np.roll(ring3d, -1, axis=0)):
        n = max(1, int(np.ceil(np.linalg.norm(b - a) / step)))
        out.append(a + (b - a) * np.linspace(0, 1, n, endpoint=False)[:, None])
    return np.vstack(out)


def sample_quad(corners, step):
    """Fill a planar quad (4,3) with a grid of points (for point-cloud viewers)."""
    a, b, _, d = corners
    nu = max(2, int(np.linalg.norm(b - a) / step) + 1)
    nv = max(2, int(np.linalg.norm(d - a) / step) + 1)
    s, t = np.meshgrid(np.linspace(0, 1, nu), np.linspace(0, 1, nv))
    return a + s.reshape(-1, 1) * (b - a) + t.reshape(-1, 1) * (d - a)


def write_overlay(path, cloud_pts, cloud_rgb, panels3d, outline3d, step=0.03):
    """One PLY: the scan (dimmed) + panels in blue + roof outline in red."""
    import open3d as o3d

    pts = [np.asarray(cloud_pts)]
    rgb = [np.asarray(cloud_rgb) * 0.6 if cloud_rgb is not None and len(cloud_rgb)
           else np.full((len(cloud_pts), 3), 0.6)]
    for c in panels3d:
        q = sample_quad(c, step)
        pts.append(q)
        rgb.append(np.tile([0.1, 0.3, 0.95], (len(q), 1)))
    if outline3d is not None:
        r = sample_ring(outline3d, step / 2)
        pts.append(r)
        rgb.append(np.tile([1.0, 0.1, 0.1], (len(r), 1)))
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(np.vstack(pts)))
    pcd.colors = o3d.utility.Vector3dVector(np.clip(np.vstack(rgb), 0, 1))
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    o3d.io.write_point_cloud(str(path), pcd)
    return path


def quad_solid(corners, thickness, to_mm=1000.0):
    """CadQuery solid: planar quad (4,3) in metres extruded along its normal."""
    import cadquery as cq

    c = np.asarray(corners, float) * to_mm
    n = np.cross(c[1] - c[0], c[3] - c[0])
    n = n / np.linalg.norm(n)
    if n[2] < 0:
        n = -n
    face = cq.Face.makeFromWires(cq.Wire.makePolygon([cq.Vector(*p) for p in c], close=True))
    return cq.Solid.extrudeLinear(face, cq.Vector(*(n * thickness * to_mm)))


def export_step(path, frame, outline, panel_corners, obstacles=None,
                panel_thickness=0.035, roof_thickness=0.05, to_mm=1000.0):
    """Roof surface slab + obstacles + panel solids (from 3D corners), mm, one STEP assembly."""
    import cadquery as cq

    def plane(h):
        o = frame.origin + h * frame.n
        return cq.Plane(origin=tuple(o * to_mm), xDir=tuple(frame.u), normal=tuple(frame.n))

    def solid(poly, h0, thick):
        ring = [tuple(p * to_mm) for p in polygon_ring(poly)]
        return cq.Workplane(plane(h0)).polyline(ring).close().extrude(thick * to_mm)

    asm = cq.Assembly(name="RoofSolar")
    asm.add(solid(outline, -roof_thickness, roof_thickness), name="RoofSurface",
            color=cq.Color(0.7, 0.7, 0.7))
    if obstacles is not None and not obstacles.is_empty:
        geoms = getattr(obstacles, "geoms", [obstacles])
        for i, g in enumerate(geoms):
            asm.add(solid(g.simplify(0.05), 0, 0.5), name=f"KeepOut_{i}",
                    color=cq.Color(0.9, 0.5, 0.1))
    for i, c in enumerate(panel_corners):
        asm.add(cq.Workplane().add(quad_solid(c, panel_thickness, to_mm)), name=f"Panel_{i:03d}",
                color=cq.Color(0.1, 0.2, 0.8))
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    asm.export(str(path), exportType="STEP")
    return path


def plot_plan(path, uv, outline, usable, obstacles, panels, title=""):
    """Top-down plan PNG in roof-plane coordinates (metres)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from shapely.plotting import plot_polygon

    def draw(ax, geom, label=None, **kw):
        """Fill polygons honouring holes; one legend entry per layer."""
        for g in getattr(geom, "geoms", [geom]):
            if not g.is_empty and g.geom_type == "Polygon":
                plot_polygon(g, ax=ax, add_points=False, label=label, **kw)
                label = None

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.scatter(uv[:, 0], uv[:, 1], s=0.3, c="0.75", label="roof points")
    draw(ax, outline, facecolor="none", edgecolor="red", linewidth=1.5, label="roof outline")
    draw(ax, usable, facecolor="none", edgecolor="green", linestyle="--", linewidth=1,
         label="usable (setback)")
    if obstacles is not None:
        draw(ax, obstacles, facecolor="orange", alpha=0.4, edgecolor="darkorange", label="keep-out")
    for i, p in enumerate(panels):
        draw(ax, p, facecolor="royalblue", edgecolor="navy", alpha=0.85,
             label="panel" if i == 0 else None)
    ax.set_aspect("equal")
    ax.set_xlabel("m")
    ax.set_ylabel("m")
    ax.set_title(title)
    ax.legend(loc="upper right", fontsize=8)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return path
