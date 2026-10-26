/** Pure helpers for the benchmark annotation editor (format `badmintonvision-annotations/1`). */

export type Box = [number, number, number, number];

export interface AnnPerson {
  id: string;
  bbox: Box;
  ground_contact: [number, number] | null;
  occluded?: boolean;
}

export interface AnnFrame {
  t: number;
  people: AnnPerson[];
  reviewed?: boolean;
}

export interface AnnotationDoc {
  format: "badmintonvision-annotations/1";
  source: string;
  target_id: string | null;
  frames: AnnFrame[];
  unreviewed_frames?: AnnFrame[];
  calibration_points: { id: string; x: number; y: number }[];
  court_positions: { t: number; court_xy: [number, number] }[];
  strokes: { t: number; shot_type?: string | null; stroke_side?: string | null }[];
  [k: string]: unknown;
}

export const FORMAT = "badmintonvision-annotations/1" as const;

/** Box from two drag corners, clamped to the image; null when too small to be a person. */
export function boxFromDrag(a: [number, number], b: [number, number], w: number, h: number, minSize = 6): Box | null {
  const x1 = Math.max(0, Math.min(a[0], b[0]));
  const y1 = Math.max(0, Math.min(a[1], b[1]));
  const x2 = Math.min(w, Math.max(a[0], b[0]));
  const y2 = Math.min(h, Math.max(a[1], b[1]));
  if (x2 - x1 < minSize || y2 - y1 < minSize) return null;
  return [round1(x1), round1(y1), round1(x2), round1(y2)];
}

/** Map a client (screen) point to image pixels for an element showing a w×h image stretched to its rect. */
export function clientToImage(clientX: number, clientY: number, rect: { left: number; top: number; width: number; height: number }, w: number, h: number): [number, number] {
  return [round1(((clientX - rect.left) / rect.width) * w), round1(((clientY - rect.top) / rect.height) * h)];
}

/** Index of the smallest box containing the point (smallest wins when boxes overlap), or -1. */
export function hitBox(people: AnnPerson[], p: [number, number]): number {
  let best = -1;
  let bestArea = Infinity;
  people.forEach((q, i) => {
    const [x1, y1, x2, y2] = q.bbox;
    if (p[0] >= x1 && p[0] <= x2 && p[1] >= y1 && p[1] <= y2) {
      const area = (x2 - x1) * (y2 - y1);
      if (area < bestArea) {
        bestArea = area;
        best = i;
      }
    }
  });
  return best;
}

/** Next unused id of the form P1, P2, … across the whole document. */
export function nextPersonId(doc: AnnotationDoc): string {
  const used = new Set<string>();
  for (const f of [...doc.frames, ...(doc.unreviewed_frames ?? [])]) for (const p of f.people) used.add(p.id);
  let k = 1;
  while (used.has(`P${k}`)) k++;
  return `P${k}`;
}

/** Normalise any loaded document: all frames in one sorted list, each carrying its reviewed flag. */
export function normaliseDoc(raw: Record<string, unknown>): AnnotationDoc {
  const d = raw as Partial<AnnotationDoc>;
  const reviewed = (d.frames ?? []).map((f) => ({ ...f, reviewed: f.reviewed ?? d.source !== "pre-annotation" }));
  const pending = (d.unreviewed_frames ?? []).map((f) => ({ ...f, reviewed: false }));
  return {
    ...(raw as object),
    format: FORMAT,
    source: d.source ?? "manual",
    target_id: d.target_id ?? null,
    frames: [...reviewed, ...pending].sort((a, b) => a.t - b.t),
    calibration_points: d.calibration_points ?? [],
    court_positions: d.court_positions ?? [],
    strokes: d.strokes ?? [],
  } as AnnotationDoc;
}

/**
 * Document to save: only frames a human marked reviewed are ground truth (`frames`); the rest are
 * kept in `unreviewed_frames`, which the evaluator ignores, so an uncorrected template can never
 * be scored against itself.
 */
export function toSavedDoc(doc: AnnotationDoc): AnnotationDoc {
  const reviewed = doc.frames.filter((f) => f.reviewed);
  const pending = doc.frames.filter((f) => !f.reviewed);
  const { warning: _w, ...rest } = doc as AnnotationDoc & { warning?: unknown };
  return {
    ...rest,
    source: "manual",
    frames: reviewed.map(({ reviewed: _r, ...f }) => f),
    unreviewed_frames: pending.map(({ reviewed: _r, ...f }) => f),
  };
}

function round1(v: number): number {
  return Math.round(v * 10) / 10;
}
