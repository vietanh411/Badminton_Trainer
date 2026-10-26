/** 3×3 homography helpers (row-major, matching numpy's `H.tolist()`). */
export type Mat3 = number[][];
export type Pt = [number, number];

export function project(H: Mat3, p: Pt): Pt | null {
  const r = projectW(H, p);
  return r ? r.pt : null;
}

/** Projection that also reports the homogeneous w (w > 0 = in front of the camera / horizon). */
export function projectW(H: Mat3, p: Pt): { pt: Pt; w: number } | null {
  const [x, y] = p;
  const w = H[2][0] * x + H[2][1] * y + H[2][2];
  if (!Number.isFinite(w) || Math.abs(w) < 1e-12) return null;
  const u = (H[0][0] * x + H[0][1] * y + H[0][2]) / w;
  const v = (H[1][0] * x + H[1][1] * y + H[1][2]) / w;
  if (!Number.isFinite(u) || !Number.isFinite(v)) return null;
  return { pt: [u, v], w };
}

export function invert3(m: Mat3): Mat3 | null {
  const [a, b, c] = m[0];
  const [d, e, f] = m[1];
  const [g, h, i] = m[2];
  const A = e * i - f * h;
  const B = -(d * i - f * g);
  const C = d * h - e * g;
  const det = a * A + b * B + c * C;
  if (!Number.isFinite(det) || Math.abs(det) < 1e-18) return null;
  const inv: Mat3 = [
    [A, -(b * i - c * h), b * f - c * e],
    [B, a * i - c * g, -(a * f - c * d)],
    [C, -(a * h - b * g), a * e - b * d],
  ];
  return inv.map((row) => row.map((v) => v / det));
}

export function multiply(a: Mat3, b: Mat3): Mat3 {
  return a.map((row) => [0, 1, 2].map((j) => row[0] * b[0][j] + row[1] * b[1][j] + row[2] * b[2][j]));
}

/** Diagonal scale matrix: maps (x, y) → (sx·x, sy·y). */
export function scaleMat(sx: number, sy: number): Mat3 {
  return [
    [sx, 0, 0],
    [0, sy, 0],
    [0, 0, 1],
  ];
}

/**
 * Sample a court-space segment (metres) and project it into image pixels, splitting where the line
 * crosses behind the camera so it is never drawn through the horizon.
 */
export function projectSegment(courtToImage: Mat3, a: Pt, b: Pt, n = 24): Pt[][] {
  const runs: Pt[][] = [];
  let cur: Pt[] = [];
  for (let k = 0; k <= n; k++) {
    const t = k / n;
    const r = projectW(courtToImage, [a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t]);
    if (r && r.w > 0) cur.push(r.pt);
    else if (cur.length) {
      runs.push(cur);
      cur = [];
    }
  }
  if (cur.length) runs.push(cur);
  return runs.filter((r) => r.length > 1);
}
