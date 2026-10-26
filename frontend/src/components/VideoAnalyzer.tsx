import { Pause, Play, Rewind } from "lucide-react";
import { useCallback, useEffect, useMemo, useRef, useState, type MutableRefObject } from "react";
import { invert3, multiply, projectSegment, scaleMat, type Mat3 } from "../lib/homography";
import { nearestIndex, windowBounds, windowsToLoad } from "../lib/timeindex";
import { api, mediaUrl } from "../services/api";
import type { AnnotationFrame, Annotations, CourtGeometry, Session, Stroke, TimelineEvent } from "../types/api";
import { Timeline } from "./Timeline";
import { Segmented, Toggle } from "./ui";

// MediaPipe skeleton edges (no face mesh), same as vision/pose/landmarks.py.
const SKELETON: [number, number][] = [
  [11, 12], [11, 13], [13, 15], [12, 14], [14, 16], [11, 23], [12, 24], [23, 24],
  [23, 25], [25, 27], [27, 29], [29, 31], [27, 31], [24, 26], [26, 28], [28, 30], [30, 32], [28, 32],
  [0, 11], [0, 12],
];

interface Layers {
  boxes: boolean;
  pose: boolean;
  court: boolean;
  trail: boolean;
  ids: boolean;
  conf: boolean;
}

type VideoWithRvfc = HTMLVideoElement & {
  requestVideoFrameCallback?: (cb: (now: number, meta: { mediaTime: number }) => void) => number;
  cancelVideoFrameCallback?: (h: number) => void;
};

function contentRect(v: HTMLVideoElement): { x: number; y: number; w: number; h: number } {
  const bw = v.clientWidth;
  const bh = v.clientHeight;
  const vw = v.videoWidth || bw;
  const vh = v.videoHeight || bh;
  const s = Math.min(bw / vw, bh / vh);
  const w = vw * s;
  const h = vh * s;
  return { x: (bw - w) / 2, y: (bh - h) / 2, w, h };
}

export function VideoAnalyzer({
  session,
  court,
  events,
  strokes,
  interval,
  onInterval,
  seekRef,
  onTime,
}: {
  session: Session;
  court: CourtGeometry | null;
  events: TimelineEvent[];
  strokes: Stroke[];
  interval: [number, number] | null;
  onInterval: (iv: [number, number] | null) => void;
  seekRef?: MutableRefObject<((t: number) => void) | null>;
  onTime?: (t: number) => void;
}) {
  const videoRef = useRef<VideoWithRvfc>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const cache = useRef(new Map<number, Annotations | "loading">());
  const [loadedTick, setLoadedTick] = useState(0);
  const onTimeRef = useRef(onTime);
  onTimeRef.current = onTime;
  const [time, setTime] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [rate, setRate] = useState("1");
  const [layers, setLayers] = useState<Layers>({ boxes: true, pose: true, court: true, trail: true, ids: true, conf: false });
  const [videoError, setVideoError] = useState<string | null>(null);
  const duration = session.video.duration_s;

  // Court → full-resolution image homography.
  const courtToImage = useMemo<Mat3 | null>(() => {
    const inv = session.calibration ? invert3(session.calibration.homography) : null;
    if (!inv) return null;
    // Homographies are defined up to scale; orient so visible court points have w > 0.
    const w = inv[2][0] * 3.05 + inv[2][1] * 3.35 + inv[2][2];
    return w < 0 ? inv.map((row) => row.map((v) => -v)) : inv;
  }, [session.calibration]);

  const ensureWindows = useCallback(
    (t: number) => {
      for (const k of windowsToLoad(t, duration)) {
        if (cache.current.has(k)) continue;
        cache.current.set(k, "loading");
        const { start, end } = windowBounds(k);
        api.annotations(session.id, start, Math.min(end, duration + 1)).then(
          (a) => {
            cache.current.set(k, a);
            setLoadedTick((x) => x + 1);
          },
          () => cache.current.delete(k),
        );
      }
    },
    [session.id, duration],
  );

  const frameAt = useCallback((t: number): { frame: AnnotationFrame | null; ann: Annotations | null; trail: AnnotationFrame[] } => {
    const k = Math.max(0, Math.floor(t / 30));
    const ann = cache.current.get(k);
    if (!ann || ann === "loading") return { frame: null, ann: null, trail: [] };
    const i = nearestIndex(ann.frames, t, (f) => f.t, 0.25);
    const frame = i >= 0 ? ann.frames[i] : null;
    const trail = i >= 0 ? ann.frames.slice(Math.max(0, i - 40), i + 1).filter((f) => f.t >= t - 2.5) : [];
    return { frame, ann, trail };
  }, []);

  const draw = useCallback(
    (t: number) => {
      const v = videoRef.current;
      const c = canvasRef.current;
      if (!v || !c) return;
      const dpr = window.devicePixelRatio || 1;
      const bw = v.clientWidth;
      const bh = v.clientHeight;
      if (c.width !== Math.round(bw * dpr) || c.height !== Math.round(bh * dpr)) {
        c.width = Math.round(bw * dpr);
        c.height = Math.round(bh * dpr);
      }
      const ctx = c.getContext("2d");
      if (!ctx) return;
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      ctx.clearRect(0, 0, bw, bh);
      const r = contentRect(v);
      ensureWindows(t);

      const { frame, ann, trail } = frameAt(t);

      // Court lines from the calibration (full-resolution pixels → display). Not drawn over shots that
      // were excluded because they do not show the calibrated court view (replays, graphics, close-ups).
      if (layers.court && court && courtToImage && frame?.status !== "excluded_shot") {
        const fullToDisplay = scaleMat(r.w / session.video.width, r.h / session.video.height);
        const H = multiply(fullToDisplay, courtToImage);
        ctx.save();
        ctx.translate(r.x, r.y);
        ctx.lineWidth = 1.5;
        for (const ln of court.lines) {
          ctx.strokeStyle = ln.kind === "net" ? "rgba(251,191,36,0.75)" : "rgba(34,211,238,0.55)";
          ctx.setLineDash(ln.kind === "net" ? [6, 4] : []);
          for (const run of projectSegment(H, ln.start, ln.end)) {
            ctx.beginPath();
            run.forEach(([x, y], i) => (i ? ctx.lineTo(x, y) : ctx.moveTo(x, y)));
            ctx.stroke();
          }
        }
        ctx.restore();
      }

      if (!frame || !ann) return;
      const s = r.w / ann.analysis_width;
      const X = (x: number) => r.x + x * s;
      const Y = (y: number) => r.y + y * s;

      if (layers.trail && trail.length > 1) {
        ctx.lineWidth = 3;
        for (let i = 1; i < trail.length; i++) {
          const a = trail[i - 1].foot;
          const b = trail[i].foot;
          // Never draw across a gap or a video cut: positions on either side are unrelated.
          if (!a || !b || trail[i].t - trail[i - 1].t > 0.3 || trail[i].shot !== trail[i - 1].shot) continue;
          ctx.strokeStyle = `rgba(163,230,53,${(0.15 + 0.75 * (i / trail.length)).toFixed(2)})`;
          ctx.beginPath();
          ctx.moveTo(X(a[0]), Y(a[1]));
          ctx.lineTo(X(b[0]), Y(b[1]));
          ctx.stroke();
        }
      }

      if (layers.boxes) {
        ctx.font = "600 11px Inter, sans-serif";
        for (const p of frame.people) {
          const isTarget = p.id === frame.target;
          const [x1, y1, x2, y2] = p.b;
          ctx.lineWidth = isTarget ? 2.5 : 1.25;
          ctx.strokeStyle = isTarget ? "#22d3ee" : "rgba(201,212,229,0.6)";
          ctx.setLineDash(isTarget && frame.status === "occluded" ? [5, 3] : []);
          ctx.strokeRect(X(x1), Y(y1), (x2 - x1) * s, (y2 - y1) * s);
          ctx.setLineDash([]);
          const parts: string[] = [];
          if (layers.ids) parts.push(isTarget ? `TARGET · ID ${p.id}` : `ID ${p.id}`);
          if (layers.conf) parts.push(`${Math.round(p.s * 100)}%`);
          if (parts.length) {
            const label = parts.join(" · ");
            const w = ctx.measureText(label).width + 10;
            ctx.fillStyle = isTarget ? "rgba(34,211,238,0.9)" : "rgba(11,18,32,0.8)";
            ctx.fillRect(X(x1), Y(y1) - 18, w, 16);
            ctx.fillStyle = isTarget ? "#0b1220" : "#c9d4e5";
            ctx.fillText(label, X(x1) + 5, Y(y1) - 6);
          }
        }
      }

      if (layers.pose && frame.pose) {
        const P = frame.pose;
        ctx.lineWidth = 2;
        ctx.strokeStyle = "rgba(163,230,53,0.9)";
        for (const [a, b] of SKELETON) {
          if (P[a][2] < 0.5 || P[b][2] < 0.5) continue;
          ctx.beginPath();
          ctx.moveTo(X(P[a][0]), Y(P[a][1]));
          ctx.lineTo(X(P[b][0]), Y(P[b][1]));
          ctx.stroke();
        }
        ctx.fillStyle = "#f8fafc";
        for (const [x, y, vis] of P.slice(11)) {
          if (vis < 0.5) continue;
          ctx.beginPath();
          ctx.arc(X(x), Y(y), 2.2, 0, Math.PI * 2);
          ctx.fill();
        }
      }

      if (frame.foot) {
        ctx.fillStyle = frame.src === "bbox" ? "#fbbf24" : "#a3e635";
        ctx.beginPath();
        ctx.arc(X(frame.foot[0]), Y(frame.foot[1]), 4, 0, Math.PI * 2);
        ctx.fill();
      }

      // HUD
      const hud: string[] = [];
      if (typeof frame.speed === "number") hud.push(`${frame.speed.toFixed(2)} m/s`);
      if (frame.court) hud.push(`x ${frame.court[0].toFixed(2)} · y ${frame.court[1].toFixed(2)} m`);
      if (layers.conf) hud.push(`position: ${frame.src} (${Math.round(frame.conf * 100)}%)`);
      if (frame.target === null) hud.push("target not tracked");
      else if (frame.status !== "tracked") hud.push(frame.status);
      if (hud.length) {
        ctx.font = "500 12px 'JetBrains Mono', monospace";
        const text = hud.join("   ");
        const w = ctx.measureText(text).width + 16;
        ctx.fillStyle = "rgba(7,13,24,0.75)";
        ctx.fillRect(r.x + 10, r.y + 10, w, 24);
        ctx.fillStyle = frame.target === null ? "#f87171" : "#e2e8f0";
        ctx.fillText(text, r.x + 18, r.y + 26);
      }
    },
    [court, courtToImage, ensureWindows, frameAt, layers, session.video.width, session.video.height],
  );

  // Sync drawing to decoded frames (rVFC) with a rAF fallback.
  useEffect(() => {
    const v = videoRef.current;
    if (!v) return;
    let handle = 0;
    let raf = 0;
    let lastReport = 0;
    const report = (t: number) => {
      const now = performance.now();
      if (now - lastReport > 120) {
        lastReport = now;
        setTime(t);
        onTimeRef.current?.(t);
      }
    };
    if (v.requestVideoFrameCallback) {
      const cb = (_now: number, meta: { mediaTime: number }) => {
        draw(meta.mediaTime);
        report(meta.mediaTime);
        handle = v.requestVideoFrameCallback!(cb);
      };
      handle = v.requestVideoFrameCallback(cb);
    } else {
      const loop = () => {
        draw(v.currentTime);
        report(v.currentTime);
        raf = requestAnimationFrame(loop);
      };
      raf = requestAnimationFrame(loop);
    }
    const redraw = () => {
      draw(v.currentTime);
      setTime(v.currentTime);
      onTimeRef.current?.(v.currentTime);
    };
    v.addEventListener("seeked", redraw);
    v.addEventListener("loadeddata", redraw);
    const ro = new ResizeObserver(redraw);
    ro.observe(v);
    redraw();
    return () => {
      if (handle && v.cancelVideoFrameCallback) v.cancelVideoFrameCallback(handle);
      cancelAnimationFrame(raf);
      v.removeEventListener("seeked", redraw);
      v.removeEventListener("loadeddata", redraw);
      ro.disconnect();
    };
  }, [draw]);

  // Redraw when an annotation window arrives while paused.
  useEffect(() => {
    if (videoRef.current) draw(videoRef.current.currentTime);
  }, [loadedTick, draw]);

  const seek = useCallback((t: number) => {
    const v = videoRef.current;
    if (v) v.currentTime = Math.max(0, Math.min(t, duration));
  }, [duration]);
  useEffect(() => {
    if (seekRef) seekRef.current = seek;
  }, [seek, seekRef]);

  useEffect(() => {
    if (videoRef.current) videoRef.current.playbackRate = Number(rate);
  }, [rate]);

  const toggle = (k: keyof Layers) => (val: boolean) => setLayers((l) => ({ ...l, [k]: val }));

  return (
    <div className="space-y-3">
      <div className="relative overflow-hidden rounded-xl border border-edge bg-black">
        <video
          ref={videoRef}
          src={mediaUrl(`/api/videos/${session.video.id}/preview`)}
          className="block aspect-video w-full"
          preload="auto"
          playsInline
          muted
          onPlay={() => setPlaying(true)}
          onPause={() => setPlaying(false)}
          onError={() =>
            setVideoError(
              session.video.browser_playable || session.video.has_proxy
                ? "The video could not be loaded."
                : `This video's codec (${session.video.codec}) cannot be played by browsers. A playable preview is created when the analysis runs.`,
            )
          }
          onClick={() => {
            const v = videoRef.current;
            if (v) void (v.paused ? v.play() : v.pause());
          }}
        />
        <canvas ref={canvasRef} className="pointer-events-none absolute inset-0 h-full w-full" />
        {videoError && <div className="absolute inset-0 grid place-items-center bg-black/70 p-6 text-center text-sm text-warn">{videoError}</div>}
      </div>
      <div className="flex flex-wrap items-center gap-3">
        <button
          className="grid h-9 w-9 place-items-center rounded-full bg-accent text-navy-900 transition hover:bg-cyan-300"
          onClick={() => {
            const v = videoRef.current;
            if (v) void (v.paused ? v.play() : v.pause());
          }}
          aria-label={playing ? "Pause" : "Play"}
        >
          {playing ? <Pause className="h-4 w-4" /> : <Play className="h-4 w-4 translate-x-px" />}
        </button>
        <button className="grid h-9 w-9 place-items-center rounded-full border border-edge text-soft hover:text-accent" onClick={() => seek(0)} aria-label="Restart">
          <Rewind className="h-4 w-4" />
        </button>
        <Segmented
          value={rate}
          onChange={setRate}
          options={[
            { value: "0.25", label: "¼×" },
            { value: "0.5", label: "½×" },
            { value: "1", label: "1×" },
          ]}
        />
        <div className="ml-auto flex flex-wrap gap-x-4 gap-y-2">
          <Toggle checked={layers.boxes} onChange={toggle("boxes")} label="Boxes" />
          <Toggle checked={layers.ids} onChange={toggle("ids")} label="IDs" />
          <Toggle checked={layers.pose} onChange={toggle("pose")} label="Skeleton" />
          <Toggle checked={layers.court} onChange={toggle("court")} label="Court" />
          <Toggle checked={layers.trail} onChange={toggle("trail")} label="Trail" />
          <Toggle checked={layers.conf} onChange={toggle("conf")} label="Confidence" />
        </div>
      </div>
      <Timeline duration={duration} time={time} events={events} strokes={strokes} interval={interval} onSeek={seek} onInterval={onInterval} />
    </div>
  );
}
