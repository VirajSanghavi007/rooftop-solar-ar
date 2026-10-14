// Corner-tap placement math, no dependencies (also run by the Python tests through node).
// Model and world are y-up metres. Placement = scale, then yaw about +y (three.js
// rotation.y convention), then translate, so model corner a lands on tapped A and
// the model edge a->b points along the tapped edge A->B.

const horiz = (p, q) => Math.hypot(q[0] - p[0], q[2] - p[2]);
// angle of a horizontal direction in three.js yaw convention (rotation.y maps +x to (cos, 0, -sin))
const heading = (p, q) => Math.atan2(-(q[2] - p[2]), q[0] - p[0]);

export function rotateY([x, y, z], yaw) {
  const c = Math.cos(yaw), s = Math.sin(yaw);
  return [x * c + z * s, y, -x * s + z * c];
}

export function solvePlacement(a, b, A, B, fit = false) {
  const designLength = horiz(a, b);
  const tappedLength = horiz(A, B);
  if (designLength < 1e-6 || tappedLength < 1e-6) throw new Error("corners too close together");
  const scale = fit ? tappedLength / designLength : 1;
  const yaw = heading(A, B) - heading(a, b);
  const r = rotateY(a.map((v) => v * scale), yaw);
  return { yaw, scale, position: [A[0] - r[0], A[1] - r[1], A[2] - r[2]],
           designLength, tappedLength, errorPct: (100 * (tappedLength - designLength)) / designLength };
}

export function applyPlacement(pl, m) {
  const r = rotateY(m.map((v) => v * pl.scale), pl.yaw);
  return [r[0] + pl.position[0], r[1] + pl.position[1], r[2] + pl.position[2]];
}
