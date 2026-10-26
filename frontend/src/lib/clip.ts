export const MAX_CLIP_S = 60;

/** Clamp a requested annotated clip to the backend's limits (at most 60 s, end > start, inside the video). */
export function clipRange(start: number, end: number, duration: number): [number, number] {
  const s = Math.max(0, Math.min(start, Math.max(0, duration - 0.5)));
  let e = Math.min(duration, Math.max(end, s + 0.5));
  if (e - s > MAX_CLIP_S) e = s + MAX_CLIP_S;
  return [s, e];
}
