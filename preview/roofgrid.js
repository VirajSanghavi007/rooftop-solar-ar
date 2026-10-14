// Live panel layout inside a tapped 4-corner roof. No dependencies (tested from Python through node).
// Points are 2D [x, z] on the horizontal plane, metres. Panels keep their real size;
// only the count changes with the roof.

export const PANEL = { long: 1.722, short: 1.134, watts: 400 };

const sub = (a, b) => [a[0] - b[0], a[1] - b[1]];
const add = (a, b) => [a[0] + b[0], a[1] + b[1]];
const mul = (a, k) => [a[0] * k, a[1] * k];
const cross = (a, b) => a[0] * b[1] - a[1] * b[0];
const dot = (a, b) => a[0] * b[0] + a[1] * b[1];
const len = (a) => Math.hypot(a[0], a[1]);

export function signedArea(poly) {
  let s = 0;
  for (let i = 0; i < poly.length; i++) s += cross(poly[i], poly[(i + 1) % poly.length]);
  return s / 2;
}

// Order taps around their centroid (tap order then does not matter), starting at
// A and continuing to B. Throws if A and B are not neighbours (B diagonal to A).
export function orderCorners(pts) {
  const c = mul(pts.reduce(add, [0, 0]), 1 / pts.length);
  const idx = pts.map((p, i) => i).sort((i, j) =>
    Math.atan2(pts[i][1] - c[1], pts[i][0] - c[0]) - Math.atan2(pts[j][1] - c[1], pts[j][0] - c[0]));
  let ring = idx.slice(idx.indexOf(0)).concat(idx.slice(0, idx.indexOf(0)));
  if (ring[1] !== 1) ring = [ring[0], ...ring.slice(1).reverse()];
  if (ring[1] !== 1) throw new Error("A and B must be neighbouring corners, not opposite ones");
  return ring.map((i) => pts[i]);
}

export function isConvex(poly) {
  const n = poly.length;
  let sign = 0;
  for (let i = 0; i < n; i++) {
    const z = cross(sub(poly[(i + 1) % n], poly[i]), sub(poly[(i + 2) % n], poly[(i + 1) % n]));
    if (Math.abs(z) < 1e-12) continue;
    if (sign && Math.sign(z) !== sign) return false;
    sign = Math.sign(z);
  }
  return sign !== 0;
}

export function insideConvex(poly, p, eps = 1e-9) {
  const s = Math.sign(signedArea(poly));
  for (let i = 0; i < poly.length; i++)
    if (s * cross(sub(poly[(i + 1) % poly.length], poly[i]), sub(p, poly[i])) < -eps) return false;
  return true;
}

// Move every edge of a convex polygon inward by d; null if it collapses.
export function insetConvex(poly, d) {
  if (d <= 0) return poly.slice();
  const n = poly.length, s = Math.sign(signedArea(poly));
  const lines = poly.map((p, i) => {
    const e = sub(poly[(i + 1) % n], p), l = len(e);
    const inward = mul([-e[1] * s, e[0] * s], 1 / l);
    return { p: add(p, mul(inward, d)), e };
  });
  const out = lines.map((L, i) => {
    const M = lines[(i + n - 1) % n];
    const den = cross(M.e, L.e);
    if (Math.abs(den) < 1e-12) return L.p;
    const t = cross(sub(L.p, M.p), L.e) / den;
    return add(M.p, mul(M.e, t));
  });
  // a collapsed inset can come back turned inside out: every edge must keep its direction
  const sameDir = out.every((p, i) => dot(sub(out[(i + 1) % n], p), lines[i].e) > 1e-9);
  const ok = sameDir && Math.sign(signedArea(out)) === s && isConvex(out) && Math.abs(signedArea(out)) > 1e-6;
  return ok ? out : null;
}

// Winter-solstice noon sun altitude and the ground gap behind a tilted row.
export const winterSunAltitude = (lat) => 90 - Math.abs(lat) - 23.44;
export function rowGap(depth, tiltDeg, sunAltDeg) {
  if (!tiltDeg) return 0;
  const rise = depth * Math.sin((tiltDeg * Math.PI) / 180);
  return rise / Math.tan((Math.max(sunAltDeg, 5) * Math.PI) / 180);
}

// Best grid of real-size panels inside the roof, rows parallel to edge A->B.
// Tilted panels face edge A->B (tap A and B along the sunny side).
// Returns panels as footprints in the (s, t) frame: s along A->B, t into the roof.
export function layoutRoof(corners, { setback = 0.5, gap = 0.02, tiltDeg = 0, sunAltDeg = 45,
                                      steps = 8, panel = PANEL } = {}) {
  const roof = orderCorners(corners);
  if (!isConvex(roof)) throw new Error("the 4 corners do not make a convex roof, tap again");
  const A = roof[0];
  const u = mul(sub(roof[1], A), 1 / len(sub(roof[1], A)));
  let v = [-u[1], u[0]];
  const c = mul(roof.reduce(add, [0, 0]), 1 / 4);
  if (dot(sub(c, A), v) < 0) v = mul(v, -1);
  const toST = (p) => [dot(sub(p, A), u), dot(sub(p, A), v)];
  const toXZ = ([s, t]) => add(A, add(mul(u, s), mul(v, t)));
  const usable = insetConvex(roof, setback);
  const result = { roof, usable, u, v, origin: A, toXZ, panels: [], orientation: null,
                   rowPitch: 0, depth: 0, width: 0, tiltDeg };
  if (!usable) return result;
  const st = usable.map(toST);
  const smin = Math.min(...st.map((p) => p[0])), smax = Math.max(...st.map((p) => p[0]));
  const tmin = Math.min(...st.map((p) => p[1])), tmax = Math.max(...st.map((p) => p[1]));
  const cos = Math.cos((tiltDeg * Math.PI) / 180);
  for (const [orientation, width, depth] of [["landscape", panel.long, panel.short], ["portrait", panel.short, panel.long]]) {
    const foot = depth * cos;                                  // footprint depth on the roof
    const pitchS = width + gap;
    const pitchT = tiltDeg ? foot + rowGap(depth, tiltDeg, sunAltDeg) : depth + gap;
    for (let i = 0; i < steps; i++) for (let j = 0; j < steps; j++) {
      const panels = [];
      for (let t = tmin + (j / steps) * pitchT; t + foot <= tmax + 1e-9; t += pitchT)
        for (let s = smin + (i / steps) * pitchS; s + width <= smax + 1e-9; s += pitchS) {
          const box = [[s, t], [s + width, t], [s + width, t + foot], [s, t + foot]].map(toXZ);
          if (box.every((p) => insideConvex(usable, p))) panels.push({ s, t, width, foot, depth });
        }
      if (panels.length > result.panels.length)
        Object.assign(result, { panels, orientation, rowPitch: pitchT, depth, width });
    }
  }
  return result;
}

export function sideLengths(roof) {
  return roof.map((p, i) => len(sub(roof[(i + 1) % roof.length], p)));
}
export const area = (poly) => Math.abs(signedArea(poly));
