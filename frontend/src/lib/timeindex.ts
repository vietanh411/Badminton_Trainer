/** Time indexing for synchronised playback overlays. */

/** First index whose time is >= t (array sorted ascending). Returns length when none. */
export function lowerBound<T>(items: readonly T[], t: number, key: (x: T) => number): number {
  let lo = 0;
  let hi = items.length;
  while (lo < hi) {
    const mid = (lo + hi) >>> 1;
    if (key(items[mid]) < t) lo = mid + 1;
    else hi = mid;
  }
  return lo;
}

/** Index of the item nearest in time, or -1 when empty or further than maxGap seconds. */
export function nearestIndex<T>(items: readonly T[], t: number, key: (x: T) => number, maxGap = Infinity): number {
  if (!items.length) return -1;
  const i = lowerBound(items, t, key);
  let best = -1;
  let bestD = Infinity;
  for (const k of [i - 1, i]) {
    if (k >= 0 && k < items.length) {
      const d = Math.abs(key(items[k]) - t);
      if (d < bestD) {
        bestD = d;
        best = k;
      }
    }
  }
  return bestD <= maxGap ? best : -1;
}

export const WINDOW_S = 30;

/** Fixed, aligned annotation windows so cache keys are stable: [k·size, (k+1)·size]. */
export function windowIndex(t: number, size = WINDOW_S): number {
  return Math.max(0, Math.floor(t / size));
}

export function windowBounds(k: number, size = WINDOW_S): { start: number; end: number } {
  return { start: k * size, end: (k + 1) * size };
}

/** Windows needed for playback at time t: the current one, plus the next when close to its end. */
export function windowsToLoad(t: number, duration: number, size = WINDOW_S, prefetchS = 10): number[] {
  const k = windowIndex(t, size);
  const out = [k];
  const { end } = windowBounds(k, size);
  if (end - t < prefetchS && end < duration) out.push(k + 1);
  return out;
}
