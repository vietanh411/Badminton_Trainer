import type { FrameMetrics, Hand, KeyFrame } from "../../shared/types";
import { drawFrameBadge, drawSkeleton } from "./draw";
import { minKnee, roundMetrics, type DetectedStroke } from "./metrics";
import { seek, type PoseSample } from "./pose";

const MAX_FRAMES = 10;
const MIN_FRAMES = 6;
const MAX_SIDE = 1024;

interface Pick {
  index: number;
  label: string;
  priority: number;
}

/** Choose the moments worth showing the coach: swing contacts first, then lunges, preparation, and overview. */
export function selectKeyFrames(
  samples: PoseSample[],
  metrics: FrameMetrics[],
  strokes: DetectedStroke[],
): { index: number; label: string }[] {
  const candidates: Pick[] = [];

  for (const s of strokes) {
    candidates.push({ index: s.sampleIndex, label: `Swing ${s.event.index}: near contact`, priority: 1 });
    if (s.prepIndex !== null) {
      candidates.push({ index: s.prepIndex, label: `Swing ${s.event.index}: preparation`, priority: 3 });
    }
  }

  let deepest = -1;
  metrics.forEach((m, i) => {
    const k = minKnee(m);
    if (k !== null && (deepest < 0 || k < minKnee(metrics[deepest])!)) deepest = i;
  });
  if (deepest >= 0) candidates.push({ index: deepest, label: "Deepest knee bend / lunge", priority: 2 });

  const chosen: Pick[] = [];
  const farEnough = (i: number, gap: number) => chosen.every((c) => Math.abs(samples[c.index].t - samples[i].t) >= gap);

  for (const c of candidates.sort((a, b) => a.priority - b.priority)) {
    if (chosen.length >= MAX_FRAMES) break;
    if (farEnough(c.index, 0.15)) chosen.push(c);
  }

  // Top up with evenly spaced frames so the coach sees the clip as a whole.
  const withPose = samples.map((s, i) => (s.image ? i : -1)).filter((i) => i >= 0);
  const want = Math.max(MIN_FRAMES - chosen.length, chosen.length < 4 ? 2 : 0);
  for (let n = 0; n < want && withPose.length; n++) {
    const i = withPose[Math.floor(((n + 0.5) / want) * withPose.length)];
    if (chosen.length < MAX_FRAMES && farEnough(i, 0.4)) chosen.push({ index: i, label: "Overview", priority: 4 });
  }

  return chosen.sort((a, b) => a.index - b.index).map(({ index, label }) => ({ index, label }));
}

export interface CapturedFrame extends KeyFrame {
  dataUrl: string;
}

export async function captureKeyFrames(
  video: HTMLVideoElement,
  samples: PoseSample[],
  metrics: FrameMetrics[],
  picks: { index: number; label: string }[],
  hand: Hand,
  onProgress: (fraction: number) => void,
): Promise<CapturedFrame[]> {
  const scale = Math.min(1, MAX_SIDE / Math.max(video.videoWidth, video.videoHeight));
  const width = Math.round(video.videoWidth * scale);
  const height = Math.round(video.videoHeight * scale);
  const canvas = document.createElement("canvas");
  canvas.width = width;
  canvas.height = height;
  const ctx = canvas.getContext("2d")!;

  const frames: CapturedFrame[] = [];
  for (const [n, pick] of picks.entries()) {
    const sample = samples[pick.index];
    await seek(video, sample.t);
    ctx.drawImage(video, 0, 0, width, height);
    if (sample.image) drawSkeleton(ctx, sample.image, width, height, hand);
    const id = n + 1;
    drawFrameBadge(ctx, `#${id}  ${sample.t.toFixed(2)}s`, width, height);

    const dataUrl = canvas.toDataURL("image/jpeg", 0.85);
    frames.push({
      id,
      timeSec: Math.round(sample.t * 100) / 100,
      label: pick.label,
      imageBase64: dataUrl.slice(dataUrl.indexOf(",") + 1),
      metrics: roundMetrics(metrics[pick.index]),
      dataUrl,
    });
    onProgress((n + 1) / picks.length);
  }
  return frames;
}
