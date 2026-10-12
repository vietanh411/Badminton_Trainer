import type { Hand } from "../../shared/types";
import { LM } from "./metrics";
import type { Pt } from "./pose";

export const HIT_ARM_COLOR = "#ff8a1f";
export const BODY_COLOR = "#38d3ff";

const BODY: [number, number][] = [
  [LM.lShoulder, LM.rShoulder],
  [LM.lShoulder, LM.lHip],
  [LM.rShoulder, LM.rHip],
  [LM.lHip, LM.rHip],
  [LM.lHip, LM.lKnee],
  [LM.lKnee, LM.lAnkle],
  [LM.rHip, LM.rKnee],
  [LM.rKnee, LM.rAnkle],
  [LM.lAnkle, 31],
  [LM.rAnkle, 32],
  [LM.lAnkle, 29],
  [LM.rAnkle, 30],
];

const LEFT_ARM: [number, number][] = [
  [LM.lShoulder, LM.lElbow],
  [LM.lElbow, LM.lWrist],
];
const RIGHT_ARM: [number, number][] = [
  [LM.rShoulder, LM.rElbow],
  [LM.rElbow, LM.rWrist],
];

/** Draw a pose skeleton. Landmarks are normalised (0-1); width/height are the target size in pixels. */
export function drawSkeleton(
  ctx: CanvasRenderingContext2D,
  pts: Pt[],
  width: number,
  height: number,
  hand: Hand,
) {
  const scale = Math.max(width, height) / 900;
  const hitArm = hand === "right" ? RIGHT_ARM : LEFT_ARM;
  const otherArm = hand === "right" ? LEFT_ARM : RIGHT_ARM;

  const line = (pairs: [number, number][], color: string, lineWidth: number) => {
    ctx.strokeStyle = color;
    ctx.lineWidth = lineWidth * scale;
    ctx.lineCap = "round";
    for (const [a, b] of pairs) {
      const p = pts[a];
      const q = pts[b];
      if (!p || !q || p.v < 0.3 || q.v < 0.3) continue;
      ctx.beginPath();
      ctx.moveTo(p.x * width, p.y * height);
      ctx.lineTo(q.x * width, q.y * height);
      ctx.stroke();
    }
  };

  ctx.save();
  ctx.globalAlpha = 0.85;
  line([...BODY, ...otherArm], BODY_COLOR, 3);
  line(hitArm, HIT_ARM_COLOR, 4.5);

  const joints = new Set([...BODY, ...LEFT_ARM, ...RIGHT_ARM].flat());
  const hitJoints = new Set(hitArm.flat());
  for (const j of joints) {
    const p = pts[j];
    if (!p || p.v < 0.3) continue;
    ctx.fillStyle = hitJoints.has(j) ? HIT_ARM_COLOR : "#ffffff";
    ctx.beginPath();
    ctx.arc(p.x * width, p.y * height, (hitJoints.has(j) ? 4.5 : 3) * scale, 0, Math.PI * 2);
    ctx.fill();
  }
  ctx.restore();
}

export function drawFrameBadge(ctx: CanvasRenderingContext2D, text: string, width: number, height: number) {
  const scale = Math.max(width, height) / 900;
  const pad = 8 * scale;
  ctx.save();
  ctx.font = `700 ${Math.round(22 * scale)}px system-ui, sans-serif`;
  const w = ctx.measureText(text).width + pad * 2;
  const h = 34 * scale;
  ctx.fillStyle = "rgba(0, 0, 0, 0.7)";
  ctx.beginPath();
  ctx.roundRect(pad, pad, w, h, 6 * scale);
  ctx.fill();
  ctx.fillStyle = "#ffffff";
  ctx.textBaseline = "middle";
  ctx.fillText(text, pad * 2, pad + h / 2);
  ctx.restore();
}
