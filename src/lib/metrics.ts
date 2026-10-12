import type { FrameMetrics, Hand, SessionSummary, StrokeEvent } from "../../shared/types";
import type { PoseSample, Pt } from "./pose";

// MediaPipe pose landmark indices. "Left" and "right" are the player's own sides.
export const LM = {
  nose: 0,
  lShoulder: 11,
  rShoulder: 12,
  lElbow: 13,
  rElbow: 14,
  lWrist: 15,
  rWrist: 16,
  lHip: 23,
  rHip: 24,
  lKnee: 25,
  rKnee: 26,
  lAnkle: 27,
  rAnkle: 28,
} as const;

const MIN_VISIBILITY = 0.5;

export function armIndices(hand: Hand) {
  return hand === "right"
    ? { shoulder: LM.rShoulder, elbow: LM.rElbow, wrist: LM.rWrist, hip: LM.rHip }
    : { shoulder: LM.lShoulder, elbow: LM.lElbow, wrist: LM.lWrist, hip: LM.lHip };
}

// ---------- geometry helpers ----------

const sub = (a: Pt, b: Pt) => ({ x: a.x - b.x, y: a.y - b.y, z: a.z - b.z });
const mid = (a: Pt, b: Pt): Pt => ({
  x: (a.x + b.x) / 2,
  y: (a.y + b.y) / 2,
  z: (a.z + b.z) / 2,
  v: Math.min(a.v, b.v),
});
const len3 = (p: { x: number; y: number; z: number }) => Math.hypot(p.x, p.y, p.z);
const dist3 = (a: Pt, b: Pt) => len3(sub(a, b));

function angleBetween(u: { x: number; y: number; z: number }, w: { x: number; y: number; z: number }): number {
  const d = len3(u) * len3(w);
  if (d === 0) return NaN;
  const cos = Math.max(-1, Math.min(1, (u.x * w.x + u.y * w.y + u.z * w.z) / d));
  return (Math.acos(cos) * 180) / Math.PI;
}

/** Angle at b formed by a-b-c, in degrees. */
const jointAngle = (a: Pt, b: Pt, c: Pt) => angleBetween(sub(a, b), sub(c, b));

const visible = (pts: Pt[], ...idx: number[]) => idx.every((i) => (pts[i]?.v ?? 0) >= MIN_VISIBILITY);

const round = (v: number | null, digits = 1): number | null =>
  v === null || !Number.isFinite(v) ? null : Math.round(v * 10 ** digits) / 10 ** digits;

function median(values: number[]): number | null {
  if (values.length === 0) return null;
  const s = [...values].sort((a, b) => a - b);
  const m = Math.floor(s.length / 2);
  return s.length % 2 ? s[m] : (s[m - 1] + s[m]) / 2;
}

const nonNull = (values: (number | null)[]) => values.filter((v): v is number => v !== null && Number.isFinite(v));

// ---------- smoothing ----------

/** Average each landmark with its immediate neighbours to damp pose jitter. */
export function smoothSamples(samples: PoseSample[]): PoseSample[] {
  const avg = (key: "image" | "world", i: number): Pt[] | null => {
    const centre = samples[i][key];
    if (!centre) return null;
    const group = [samples[i - 1]?.[key], centre, samples[i + 1]?.[key]].filter((g): g is Pt[] => !!g);
    return centre.map((p, j) => {
      const pts = group.map((g) => g[j]);
      return {
        x: pts.reduce((s, q) => s + q.x, 0) / pts.length,
        y: pts.reduce((s, q) => s + q.y, 0) / pts.length,
        z: pts.reduce((s, q) => s + q.z, 0) / pts.length,
        v: p.v,
      };
    });
  };
  return samples.map((s, i) => ({ t: s.t, image: avg("image", i), world: avg("world", i) }));
}

// ---------- handedness ----------

/** The hitting arm is the one whose wrist moves the most relative to the body. */
export function detectHand(samples: PoseSample[]): Hand {
  let left = 0;
  let right = 0;
  for (let i = 1; i < samples.length; i++) {
    const a = samples[i - 1].world;
    const b = samples[i].world;
    if (!a || !b) continue;
    left += dist3(a[LM.lWrist], b[LM.lWrist]);
    right += dist3(a[LM.rWrist], b[LM.rWrist]);
  }
  return left > right * 1.15 ? "left" : "right";
}

// ---------- per-frame metrics ----------

const EMPTY: FrameMetrics = {
  elbowAngle: null,
  shoulderAngle: null,
  leftKneeAngle: null,
  rightKneeAngle: null,
  stanceWidth: null,
  wristHeight: null,
  trunkLean: null,
  hipShoulderSeparation: null,
  wristSpeed: null,
  bodySpeed: null,
};

export function computeFrameMetrics(
  samples: PoseSample[],
  hand: Hand,
  width: number,
  height: number,
): FrameMetrics[] {
  const arm = armIndices(hand);
  // Image coordinates scaled to pixels so 2D angles and distances aren't distorted by aspect ratio.
  const px = (p: Pt): Pt => ({ x: p.x * width, y: p.y * height, z: 0, v: p.v });

  const torsoPx = (img: Pt[]) => {
    const sh = mid(px(img[LM.lShoulder]), px(img[LM.rShoulder]));
    const hp = mid(px(img[LM.lHip]), px(img[LM.rHip]));
    return { sh, hp, len: Math.hypot(sh.x - hp.x, sh.y - hp.y) };
  };

  const metrics = samples.map((s): FrameMetrics => {
    const w = s.world;
    const img = s.image;
    if (!w || !img) return { ...EMPTY };
    const m: FrameMetrics = { ...EMPTY };

    if (visible(img, arm.shoulder, arm.elbow, arm.wrist)) {
      m.elbowAngle = jointAngle(w[arm.shoulder], w[arm.elbow], w[arm.wrist]);
    }
    if (visible(img, arm.hip, arm.shoulder, arm.elbow)) {
      m.shoulderAngle = jointAngle(w[arm.hip], w[arm.shoulder], w[arm.elbow]);
    }
    if (visible(img, LM.lHip, LM.lKnee, LM.lAnkle)) {
      m.leftKneeAngle = jointAngle(w[LM.lHip], w[LM.lKnee], w[LM.lAnkle]);
    }
    if (visible(img, LM.rHip, LM.rKnee, LM.rAnkle)) {
      m.rightKneeAngle = jointAngle(w[LM.rHip], w[LM.rKnee], w[LM.rAnkle]);
    }
    if (visible(img, LM.lAnkle, LM.rAnkle, LM.lShoulder, LM.rShoulder)) {
      const shoulders = dist3(w[LM.lShoulder], w[LM.rShoulder]);
      if (shoulders > 0) m.stanceWidth = dist3(w[LM.lAnkle], w[LM.rAnkle]) / shoulders;
    }

    const torsoVisible = visible(img, LM.lShoulder, LM.rShoulder, LM.lHip, LM.rHip);
    if (torsoVisible) {
      const t = torsoPx(img);
      if (t.len > 0) {
        if (visible(img, arm.wrist, LM.nose)) {
          // Image y grows downward, so nose - wrist is positive when the wrist is higher.
          m.wristHeight = (px(img[LM.nose]).y - px(img[arm.wrist]).y) / t.len;
        }
        m.trunkLean = (Math.atan2(Math.abs(t.sh.x - t.hp.x), t.hp.y - t.sh.y) * 180) / Math.PI;
      }
      // Seen from above (x-z plane): how far the shoulders are rotated relative to the hips.
      const shAng = Math.atan2(w[LM.rShoulder].z - w[LM.lShoulder].z, w[LM.rShoulder].x - w[LM.lShoulder].x);
      const hipAng = Math.atan2(w[LM.rHip].z - w[LM.lHip].z, w[LM.rHip].x - w[LM.lHip].x);
      let diff = ((shAng - hipAng) * 180) / Math.PI;
      diff = ((diff + 540) % 360) - 180;
      m.hipShoulderSeparation = Math.abs(diff);
    }
    return m;
  });

  // Speeds use central differences between neighbouring samples.
  for (let i = 0; i < samples.length; i++) {
    const prev = samples[Math.max(0, i - 1)];
    const next = samples[Math.min(samples.length - 1, i + 1)];
    const dt = next.t - prev.t;
    if (dt <= 0 || !prev.world || !next.world || !prev.image || !next.image) continue;
    if (visible(prev.image, arm.wrist) && visible(next.image, arm.wrist)) {
      metrics[i].wristSpeed = dist3(prev.world[arm.wrist], next.world[arm.wrist]) / dt;
    }
    const cur = samples[i].image;
    if (cur && visible(cur, LM.lShoulder, LM.rShoulder, LM.lHip, LM.rHip)) {
      const a = torsoPx(prev.image);
      const b = torsoPx(next.image);
      const torso = torsoPx(cur).len;
      if (torso > 0) metrics[i].bodySpeed = Math.hypot(b.hp.x - a.hp.x, b.hp.y - a.hp.y) / torso / dt;
    }
  }

  // Light smoothing on wrist speed so a single noisy sample doesn't look like a swing.
  const raw = metrics.map((m) => m.wristSpeed);
  metrics.forEach((m, i) => {
    const window = nonNull([raw[i - 1] ?? null, raw[i], raw[i + 1] ?? null]);
    if (raw[i] !== null && window.length) m.wristSpeed = window.reduce((a, b) => a + b, 0) / window.length;
  });

  return metrics;
}

export function roundMetrics(m: FrameMetrics): FrameMetrics {
  return {
    elbowAngle: round(m.elbowAngle, 0),
    shoulderAngle: round(m.shoulderAngle, 0),
    leftKneeAngle: round(m.leftKneeAngle, 0),
    rightKneeAngle: round(m.rightKneeAngle, 0),
    stanceWidth: round(m.stanceWidth, 2),
    wristHeight: round(m.wristHeight, 2),
    trunkLean: round(m.trunkLean, 0),
    hipShoulderSeparation: round(m.hipShoulderSeparation, 0),
    wristSpeed: round(m.wristSpeed, 1),
    bodySpeed: round(m.bodySpeed, 1),
  };
}

export const minKnee = (m: FrameMetrics) => {
  const k = nonNull([m.leftKneeAngle, m.rightKneeAngle]);
  return k.length ? Math.min(...k) : null;
};

// ---------- swing detection ----------

export interface DetectedStroke {
  sampleIndex: number;
  prepIndex: number | null;
  event: StrokeEvent;
}

const nearestIndex = (samples: PoseSample[], t: number) => {
  let best = 0;
  for (let i = 1; i < samples.length; i++) {
    if (Math.abs(samples[i].t - t) < Math.abs(samples[best].t - t)) best = i;
  }
  return best;
};

/** Swings show up as clear peaks in hitting-wrist speed relative to the hips. */
export function detectStrokes(samples: PoseSample[], metrics: FrameMetrics[], maxStrokes = 6): DetectedStroke[] {
  const speed = metrics.map((m) => m.wristSpeed);
  const valid = nonNull(speed);
  if (valid.length < 5) return [];
  const med = median(valid) ?? 0;
  const max = Math.max(...valid);
  const threshold = Math.max(med * 2, max * 0.45, 1.0);

  let peaks: number[] = [];
  for (let i = 1; i < speed.length - 1; i++) {
    const s = speed[i];
    if (s === null || s < threshold) continue;
    if (s >= (speed[i - 1] ?? 0) && s >= (speed[i + 1] ?? 0)) peaks.push(i);
  }

  // Keep the strongest peak within any 0.7 s window.
  peaks.sort((a, b) => speed[b]! - speed[a]!);
  const kept: number[] = [];
  for (const p of peaks) {
    if (kept.every((k) => Math.abs(samples[k].t - samples[p].t) >= 0.7)) kept.push(p);
  }
  peaks = kept.slice(0, maxStrokes).sort((a, b) => a - b);

  return peaks.map((i, n) => {
    const t = samples[i].t;
    const prepIndex = t >= 0.3 ? nearestIndex(samples, t - 0.35) : null;
    const around = metrics
      .filter((_, j) => Math.abs(samples[j].t - t) <= 0.5)
      .map((m) => m.hipShoulderSeparation);
    const seps = nonNull(around);
    return {
      sampleIndex: i,
      prepIndex: prepIndex !== null && prepIndex !== i ? prepIndex : null,
      event: {
        index: n + 1,
        timeSec: round(t, 2)!,
        peakWristSpeed: round(speed[i], 1)!,
        contact: roundMetrics(metrics[i]),
        prep: prepIndex !== null && prepIndex !== i ? roundMetrics(metrics[prepIndex]) : null,
        rotationRange: seps.length >= 2 ? round(Math.max(...seps) - Math.min(...seps), 0) : null,
      },
    };
  });
}

// ---------- whole-clip summary ----------

export function summarise(
  samples: PoseSample[],
  metrics: FrameMetrics[],
  strokes: DetectedStroke[],
  info: { durationSec: number; width: number; height: number; fps: number; hand: Hand; handAutoDetected: boolean },
): SessionSummary {
  const knees = nonNull(metrics.flatMap((m) => [m.leftKneeAngle, m.rightKneeAngle]));
  const stance = nonNull(metrics.map((m) => m.stanceWidth));
  const lean = nonNull(metrics.map((m) => m.trunkLean));
  const body = nonNull(metrics.map((m) => m.bodySpeed));

  const stanceFrames = metrics.filter((m) => minKnee(m) !== null && m.stanceWidth !== null);
  const athletic = stanceFrames.filter((m) => minKnee(m)! < 160 && m.stanceWidth! >= 1.0).length;

  // Integrate body speed over time for distance travelled.
  let travel = 0;
  for (let i = 1; i < samples.length; i++) {
    const v = metrics[i].bodySpeed;
    if (v !== null) travel += v * (samples[i].t - samples[i - 1].t);
  }

  return {
    durationSec: round(info.durationSec, 1)!,
    width: info.width,
    height: info.height,
    samples: samples.length,
    sampleFps: info.fps,
    poseCoverage: round(samples.filter((s) => s.image).length / Math.max(samples.length, 1), 2)!,
    hand: info.hand,
    handAutoDetected: info.handAutoDetected,
    strokes: strokes.map((s) => s.event),
    posture: {
      medianKneeAngle: round(median(knees), 0),
      minKneeAngle: knees.length ? round(Math.min(...knees), 0) : null,
      medianStanceWidth: round(median(stance), 2),
      maxStanceWidth: stance.length ? round(Math.max(...stance), 2) : null,
      athleticStanceRatio: stanceFrames.length ? round(athletic / stanceFrames.length, 2) : null,
      medianTrunkLean: round(median(lean), 0),
    },
    movement: {
      totalTravelTorsoLengths: body.length ? round(travel, 1) : null,
      peakBodySpeed: body.length ? round(Math.max(...body), 1) : null,
      activeRatio: body.length ? round(body.filter((v) => v > 0.5).length / body.length, 2) : null,
    },
  };
}
