/** Heatmap colour scale shared with the backend's Matplotlib exports (vision/visualization/court_plot.py). */
export type RGBA = [number, number, number, number];

export const HEAT_STOPS: { t: number; c: RGBA }[] = [
  { t: 0.0, c: [15, 26, 46, 0] },
  { t: 0.15, c: [13, 89, 140, 0.55] },
  { t: 0.45, c: [34, 211, 238, 0.8] },
  { t: 0.75, c: [163, 230, 53, 0.9] },
  { t: 1.0, c: [252, 242, 140, 1] },
];

export function heatColor(t: number): RGBA {
  const first = HEAT_STOPS[0].c;
  const last = HEAT_STOPS[HEAT_STOPS.length - 1].c;
  if (!Number.isFinite(t) || t <= 0) return [...first] as RGBA;
  if (t >= 1) return [...last] as RGBA;
  for (let i = 1; i < HEAT_STOPS.length; i++) {
    const a = HEAT_STOPS[i - 1];
    const b = HEAT_STOPS[i];
    if (t <= b.t) {
      const f = (t - a.t) / (b.t - a.t);
      return [0, 1, 2, 3].map((k) => a.c[k] + (b.c[k] - a.c[k]) * f) as RGBA;
    }
  }
  return [...last] as RGBA;
}

export function cssGradient(direction = "to right"): string {
  const parts = HEAT_STOPS.map(({ t, c }) => `rgba(${c[0]},${c[1]},${c[2]},${Math.max(c[3], 0.08)}) ${t * 100}%`);
  return `linear-gradient(${direction}, ${parts.join(", ")})`;
}

/**
 * RGBA pixels for a row-major grid whose row 0 is the smallest y. With `flipY` the last row comes
 * first so that, drawn top-down, the far end of the court (large y) is at the top.
 */
export function gridToRgba(values: number[][], vmax: number, opacity = 1, flipY = true): Uint8ClampedArray {
  const ny = values.length;
  const nx = ny ? values[0].length : 0;
  const out = new Uint8ClampedArray(nx * ny * 4);
  const scale = vmax > 0 ? 1 / vmax : 0;
  for (let j = 0; j < ny; j++) {
    const row = values[flipY ? ny - 1 - j : j];
    for (let i = 0; i < nx; i++) {
      const [r, g, b, a] = heatColor(row[i] * scale);
      const k = (j * nx + i) * 4;
      out[k] = r;
      out[k + 1] = g;
      out[k + 2] = b;
      out[k + 3] = Math.round(a * opacity * 255);
    }
  }
  return out;
}
