import { useEffect, useState, type MouseEvent, type RefObject } from "react";
import type { FrameMetrics } from "../../shared/types";
import type { CapturedFrame } from "../lib/keyframes";
import { minKnee, type DetectedStroke } from "../lib/metrics";
import type { PoseSample } from "../lib/pose";

interface Props {
  samples: PoseSample[];
  metrics: FrameMetrics[];
  strokes: DetectedStroke[];
  frames: CapturedFrame[];
  duration: number;
  videoRef: RefObject<HTMLVideoElement | null>;
  onSeek: (t: number) => void;
}

const W = 600;
const H = 150;
const PAD = { top: 18, bottom: 22, left: 8, right: 8 };

function path(points: ([number, number] | null)[]): string {
  let d = "";
  let pen = false;
  for (const p of points) {
    if (!p) {
      pen = false;
      continue;
    }
    d += `${pen ? "L" : "M"}${p[0].toFixed(1)},${p[1].toFixed(1)}`;
    pen = true;
  }
  return d;
}

/** Racket-arm speed and knee bend over time, with detected swings and key frames marked. */
export function TimelineChart({ samples, metrics, strokes, frames, duration, videoRef, onSeek }: Props) {
  const [playhead, setPlayhead] = useState(0);

  useEffect(() => {
    const video = videoRef.current;
    if (!video) return;
    const update = () => setPlayhead(video.currentTime);
    video.addEventListener("timeupdate", update);
    video.addEventListener("seeked", update);
    return () => {
      video.removeEventListener("timeupdate", update);
      video.removeEventListener("seeked", update);
    };
  }, [videoRef]);

  const x = (t: number) => PAD.left + (t / Math.max(duration, 0.001)) * (W - PAD.left - PAD.right);
  const plotH = H - PAD.top - PAD.bottom;

  const speeds = metrics.map((m) => m.wristSpeed ?? 0);
  const maxSpeed = Math.max(...speeds, 0.1);
  const speedPath = path(
    metrics.map((m, i) =>
      m.wristSpeed === null ? null : [x(samples[i].t), PAD.top + plotH * (1 - m.wristSpeed / maxSpeed)],
    ),
  );

  // Knee bend = 180 - knee angle, so the line rises as the player gets lower.
  const kneePath = path(
    metrics.map((m, i) => {
      const k = minKnee(m);
      if (k === null) return null;
      const bend = Math.min(Math.max(180 - k, 0), 100) / 100;
      return [x(samples[i].t), PAD.top + plotH * (1 - bend)];
    }),
  );

  const seekFromEvent = (e: MouseEvent<SVGSVGElement>) => {
    const rect = e.currentTarget.getBoundingClientRect();
    const fx = ((e.clientX - rect.left) / rect.width) * W;
    const t = ((fx - PAD.left) / (W - PAD.left - PAD.right)) * duration;
    onSeek(Math.min(Math.max(t, 0), duration));
  };

  return (
    <figure className="timeline">
      <figcaption>
        <span className="legend">
          <i style={{ background: "var(--hit)" }} /> Racket-arm speed
        </span>
        <span className="legend">
          <i style={{ background: "var(--body)" }} /> Knee bend
        </span>
        <span className="legend">
          <i className="swing-mark" /> Swing
        </span>
      </figcaption>
      <svg
        viewBox={`0 0 ${W} ${H}`}
        onClick={seekFromEvent}
        role="img"
        aria-label="Timeline of racket-arm speed and knee bend. Click to jump to a moment."
      >
        <line x1={PAD.left} x2={W - PAD.right} y1={H - PAD.bottom} y2={H - PAD.bottom} className="axis" />
        {strokes.map((s) => (
          <line
            key={s.event.index}
            x1={x(s.event.timeSec)}
            x2={x(s.event.timeSec)}
            y1={PAD.top - 6}
            y2={H - PAD.bottom}
            className="swing-line"
          />
        ))}
        <path d={kneePath} className="series knee" />
        <path d={speedPath} className="series speed" />
        {frames.map((f) => (
          <g
            key={f.id}
            className="frame-marker"
            onClick={(e) => {
              e.stopPropagation();
              onSeek(f.timeSec);
            }}
          >
            <circle cx={x(f.timeSec)} cy={H - PAD.bottom} r={8} />
            <text x={x(f.timeSec)} y={H - PAD.bottom + 3.5}>
              {f.id}
            </text>
          </g>
        ))}
        <line x1={x(playhead)} x2={x(playhead)} y1={4} y2={H - PAD.bottom} className="playhead" />
      </svg>
    </figure>
  );
}
