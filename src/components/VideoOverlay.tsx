import { useEffect, useRef, useState, type RefObject } from "react";
import type { Hand } from "../../shared/types";
import { drawSkeleton } from "../lib/draw";
import type { PoseSample } from "../lib/pose";

interface Props {
  videoUrl: string;
  videoRef: RefObject<HTMLVideoElement | null>;
  samples: PoseSample[];
  hand: Hand;
}

function nearestSample(samples: PoseSample[], t: number): PoseSample | null {
  let lo = 0;
  let hi = samples.length - 1;
  while (lo < hi) {
    const m = (lo + hi) >> 1;
    if (samples[m].t < t) lo = m + 1;
    else hi = m;
  }
  const candidates = [samples[lo - 1], samples[lo]].filter(Boolean);
  return candidates.sort((a, b) => Math.abs(a.t - t) - Math.abs(b.t - t))[0] ?? null;
}

/** The original video with the tracked skeleton drawn on top. */
export function VideoOverlay({ videoUrl, videoRef, samples, hand }: Props) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [showSkeleton, setShowSkeleton] = useState(true);
  const [slowMo, setSlowMo] = useState(false);

  useEffect(() => {
    const video = videoRef.current;
    if (video) video.playbackRate = slowMo ? 0.25 : 1;
  }, [slowMo, videoRef]);

  useEffect(() => {
    let raf = 0;
    let lastKey = "";
    const maxGap = samples.length > 1 ? (samples[1].t - samples[0].t) * 1.5 : 0.2;

    const draw = () => {
      raf = requestAnimationFrame(draw);
      const video = videoRef.current;
      const canvas = canvasRef.current;
      if (!video || !canvas) return;

      const dpr = window.devicePixelRatio || 1;
      const w = Math.round(video.clientWidth * dpr);
      const h = Math.round(video.clientHeight * dpr);
      const key = `${video.currentTime}|${w}|${h}|${showSkeleton}`;
      if (key === lastKey) return;
      lastKey = key;

      if (canvas.width !== w || canvas.height !== h) {
        canvas.width = w;
        canvas.height = h;
      }
      const ctx = canvas.getContext("2d")!;
      ctx.clearRect(0, 0, w, h);
      if (!showSkeleton) return;
      const sample = nearestSample(samples, video.currentTime);
      if (!sample?.image || Math.abs(sample.t - video.currentTime) > maxGap || !video.videoWidth) return;

      // The picture is letterboxed inside the element when its aspect ratio differs.
      const aspect = video.videoWidth / video.videoHeight;
      const cw = Math.min(w, h * aspect);
      const ch = cw / aspect;
      ctx.save();
      ctx.translate((w - cw) / 2, (h - ch) / 2);
      drawSkeleton(ctx, sample.image, cw, ch, hand);
      ctx.restore();
    };
    draw();
    return () => cancelAnimationFrame(raf);
  }, [samples, hand, showSkeleton, videoRef]);

  return (
    <div className="overlay-player">
      <div className="overlay-stage">
        <video ref={videoRef} src={videoUrl} controls playsInline muted />
        <canvas ref={canvasRef} aria-hidden />
      </div>
      <div className="overlay-controls">
        <label className="toggle">
          <input type="checkbox" checked={showSkeleton} onChange={(e) => setShowSkeleton(e.target.checked)} />
          Skeleton
        </label>
        <label className="toggle">
          <input type="checkbox" checked={slowMo} onChange={(e) => setSlowMo(e.target.checked)} />
          Slow motion
        </label>
        <span className="legend">
          <i style={{ background: "var(--hit)" }} /> Racket arm
        </span>
      </div>
    </div>
  );
}
