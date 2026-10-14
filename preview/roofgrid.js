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

// ---------- general polygons (any number of corners, concave allowed) ----------

function segIntersect(p1, p2, q1, q2) {
  const d1 = cross(sub(p2, p1), sub(q1, p1)), d2 = cross(sub(p2, p1), sub(q2, p1));
  const d3 = cross(sub(q2, q1), sub(p1, q1)), d4 = cross(sub(q2, q1), sub(p2, q1));
  return d1 * d2 < 0 && d3 * d4 < 0;
}
function pointSegDist(p, a, b) {
  const ab = sub(b, a), l2 = dot(ab, ab);
  const t = l2 ? Math.max(0, Math.min(1, dot(sub(p, a), ab) / l2)) : 0;
  return len(sub(p, add(a, mul(ab, t))));
}
export function segDist(p1, p2, q1, q2) {
  if (segIntersect(p1, p2, q1, q2)) return 0;
  return Math.min(pointSegDist(p1, q1, q2), pointSegDist(p2, q1, q2), pointSegDist(q1, p1, p2), pointSegDist(q2, p1, p2));
}
const edges = (poly) => poly.map((p, i) => [p, poly[(i + 1) % poly.length]]);

export function isSimple(poly) {
  const E = edges(poly), n = E.length;
  if (n < 3 || area(poly) < 1e-9) return false;
  for (let i = 0; i < n; i++) for (let j = i + 1; j < n; j++) {
    if (j === i + 1 || (i === 0 && j === n - 1)) continue;   // neighbours share a corner
    if (segIntersect(E[i][0], E[i][1], E[j][0], E[j][1])) return false;
  }
  return true;
}

export function pointInPolygon(poly, p) {
  let inside = false;
  for (const [a, b] of edges(poly))
    if ((a[1] > p[1]) !== (b[1] > p[1]) && p[0] < a[0] + ((p[1] - a[1]) * (b[0] - a[0])) / (b[1] - a[1])) inside = !inside;
  return inside;
}

// Smallest gap between two polygons' outlines (0 if they cross).
export function polyDist(P, Q) {
  let d = Infinity;
  for (const [a, b] of edges(P)) for (const [c, e] of edges(Q)) d = Math.min(d, segDist(a, b, c, e));
  return d;
}

// Panel rectangle fits: inside the roof, >= setback from its edges, >= clearance from every obstacle.
export function rectFits(rect, roof, setback, obstacles = [], clearance = 0) {
  if (!rect.every((p) => pointInPolygon(roof, p))) return false;
  if (polyDist(rect, roof) < setback - 1e-9) return false;
  for (const ob of obstacles) {
    if (rect.some((p) => pointInPolygon(ob, p)) || ob.some((p) => pointInPolygon(rect, p))) return false;
    if (polyDist(rect, ob) < clearance - 1e-9) return false;
  }
  return true;
}

// Obstacle from 2 taps: box with that diagonal, aligned to the roof's A-B edge (u) and v.
export function boxFromDiagonal(p, q, u, v) {
  const s0 = dot(p, u), s1 = dot(q, u), t0 = dot(p, v), t1 = dot(q, v);
  const at = (s, t) => add(mul(u, s), mul(v, t));
  return [at(s0, t0), at(s1, t0), at(s1, t1), at(s0, t1)];
}

// Roof frame: s along A->B (first two taps), t into the roof.
export function roofFrame(roof) {
  const A = roof[0];
  const u = mul(sub(roof[1], A), 1 / len(sub(roof[1], A)));
  let v = [-u[1], u[0]];
  // interior is on the left of each edge for a counter-clockwise ring
  if (signedArea(roof) < 0) v = mul(v, -1);
  return { A, u, v, toST: (p) => [dot(sub(p, A), u), dot(sub(p, A), v)],
           toXZ: ([s, t]) => add(A, add(mul(u, s), mul(v, t))) };
}

// Best grid of real-size panels inside the roof, rows parallel to edge A->B.
// Corners are tapped in order around the roof (any count >= 3, concave allowed).
// Obstacles: polygons (>= 3 points) or 2-point diagonals of a box.
// Tilted panels face edge A->B (tap A and B along the sunny side).
export function layoutRoof(corners, { setback = 0.5, gap = 0.02, tiltDeg = 0, sunAltDeg = 45,
                                      steps = 8, panel = PANEL, obstacles = [], clearance = 0.3 } = {}) {
  let roof = corners.slice();
  if (roof.length < 3) throw new Error("tap at least 3 corners");
  if (!isSimple(roof) && roof.length === 4) roof = orderCorners(roof);
  if (!isSimple(roof)) throw new Error("the edges cross: tap the corners in order going around the roof");
  const F = roofFrame(roof);
  const obs = obstacles.map((o) => (o.length === 2 ? boxFromDiagonal(o[0], o[1], F.u, F.v) : o));
  const result = { roof, obstacles: obs, u: F.u, v: F.v, origin: F.A, toXZ: F.toXZ, panels: [],
                   orientation: null, rowPitch: 0, depth: 0, width: 0, tiltDeg };
  const st = roof.map(F.toST);
  const smin = Math.min(...st.map((p) => p[0])) + setback, smax = Math.max(...st.map((p) => p[0])) - setback;
  const tmin = Math.min(...st.map((p) => p[1])) + setback, tmax = Math.max(...st.map((p) => p[1])) - setback;
  const cos = Math.cos((tiltDeg * Math.PI) / 180);
  for (const [orientation, width, depth] of [["landscape", panel.long, panel.short], ["portrait", panel.short, panel.long]]) {
    const foot = depth * cos;                                  // footprint depth on the roof
    const pitchS = width + gap;
    const pitchT = tiltDeg ? foot + rowGap(depth, tiltDeg, sunAltDeg) : depth + gap;
    for (let i = 0; i < steps; i++) for (let j = 0; j < steps; j++) {
      const panels = [];
      for (let t = tmin + (j / steps) * pitchT; t + foot <= tmax + 1e-9; t += pitchT)
        for (let s = smin + (i / steps) * pitchS; s + width <= smax + 1e-9; s += pitchS) {
          const rect = [[s, t], [s + width, t], [s + width, t + foot], [s, t + foot]].map(F.toXZ);
          if (rectFits(rect, roof, setback, obs, clearance)) panels.push({ s, t, width, foot, depth });
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
