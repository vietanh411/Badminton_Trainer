import { Crosshair, RotateCcw, Save, Trash2, Wand2 } from "lucide-react";
import { useEffect, useMemo, useRef, useState } from "react";
import { useCourt } from "../hooks/useStatic";
import { invert3, projectSegment } from "../lib/homography";
import { fmtNum, fmtTime } from "../lib/format";
import { api, mediaUrl } from "../services/api";
import type { Calibration, Session } from "../types/api";
import { CourtSvg, toSvg } from "./CourtSvg";
import { Badge, Button, Card, cx, ErrorBox, InfoTip, Notice } from "./ui";

interface Pt {
  id: string;
  x: number; // full-resolution pixels
  y: number;
}

const SUGGESTED = [
  "near_baseline__left_doubles",
  "near_baseline__right_doubles",
  "near_short_service__left_doubles",
  "near_short_service__right_doubles",
  "far_short_service__left_doubles",
  "far_short_service__right_doubles",
  "far_baseline__left_doubles",
  "far_baseline__right_doubles",
];

const QUALITY_TONE = { good: "good", fair: "warn", poor: "bad" } as const;

export function CalibrationEditor({ session, onSaved }: { session: Session; onSaved: (c: Calibration | null) => void }) {
  const court = useCourt();
  const W = session.video.width;
  const H = session.video.height;
  const existing = session.calibration;
  const [t, setT] = useState(existing?.frame_time_s ?? Math.min(1, session.video.duration_s / 2));
  const [frameT, setFrameT] = useState(t);
  const [points, setPoints] = useState<Pt[]>(
    existing ? existing.point_ids.map((id, i) => ({ id, x: existing.image_points[i][0], y: existing.image_points[i][1] })) : [],
  );
  const [active, setActive] = useState<string>(SUGGESTED.find((k) => !existing?.point_ids.includes(k)) ?? SUGGESTED[0]);
  const [result, setResult] = useState<Calibration | null>(existing);
  const [fitError, setFitError] = useState<string | null>(null);
  const [busy, setBusy] = useState<"auto" | "save" | "reset" | null>(null);
  const [message, setMessage] = useState<string | null>(null);
  const [error, setError] = useState<unknown>(null);
  const [dirty, setDirty] = useState(false);
  const dragging = useRef<string | null>(null);
  const svgRef = useRef<SVGSVGElement>(null);

  // Debounced dry-run validation whenever the points change.
  useEffect(() => {
    if (!dirty) return;
    if (points.length < 4) {
      setResult(null);
      setFitError(points.length ? `Place ${4 - points.length} more landmark(s) — at least 4 are needed (6+ recommended).` : null);
      return;
    }
    const id = window.setTimeout(() => {
      api
        .calibrate(session.id, { points, image_width: W, image_height: H, frame_time_s: frameT, dry_run: true })
        .then((r) => {
          setResult(r);
          setFitError(null);
        })
        .catch((e: unknown) => {
          setResult(null);
          setFitError(e instanceof Error ? e.message : String(e));
        });
    }, 300);
    return () => window.clearTimeout(id);
  }, [points, dirty, session.id, W, H, frameT]);

  const overlay = useMemo(() => {
    if (!result || !court) return [];
    const inv = invert3(result.homography);
    if (!inv) return [];
    const w = inv[2][0] * 3.05 + inv[2][1] * 3.35 + inv[2][2];
    const Hc = w < 0 ? inv.map((r) => r.map((v) => -v)) : inv;
    return court.lines.map((ln) => ({ ln, runs: projectSegment(Hc, ln.start, ln.end, 40) }));
  }, [result, court]);

  const labelOf = (id: string) => court?.keypoints.find((k) => k.id === id)?.label ?? id;

  function toImage(e: { clientX: number; clientY: number }): [number, number] {
    const r = svgRef.current!.getBoundingClientRect();
    return [((e.clientX - r.left) / r.width) * W, ((e.clientY - r.top) / r.height) * H];
  }

  function place(x: number, y: number) {
    setDirty(true);
    setPoints((ps) => [...ps.filter((p) => p.id !== active), { id: active, x, y }]);
    const used = new Set([...points.map((p) => p.id), active]);
    const next = SUGGESTED.find((k) => !used.has(k));
    if (next) setActive(next);
  }

  async function autoDetect() {
    setBusy("auto");
    setMessage(null);
    setError(null);
    try {
      const r = await api.autoCalibrate(session.id, frameT);
      if (!r.success || !r.points) {
        setMessage(`Automatic detection did not find a reliable court: ${r.message}. Place the landmarks manually.`);
      } else {
        const sx = W / (r.image_width ?? W);
        const sy = H / (r.image_height ?? H);
        setPoints(r.points.map((p) => ({ id: p.id, x: p.x * sx, y: p.y * sy })));
        setDirty(true);
        setMessage(`Court detected (${Math.round(r.score * 100)}% line match). Check every point against the lines before saving.`);
      }
    } catch (e) {
      setError(e);
    } finally {
      setBusy(null);
    }
  }

  async function save() {
    setBusy("save");
    setError(null);
    try {
      const r = await api.calibrate(session.id, { points, image_width: W, image_height: H, frame_time_s: frameT, dry_run: false });
      setResult(r);
      setDirty(false);
      setMessage("Calibration saved.");
      onSaved(r);
    } catch (e) {
      setError(e);
    } finally {
      setBusy(null);
    }
  }

  async function reset() {
    setBusy("reset");
    try {
      await api.resetCalibration(session.id);
      setPoints([]);
      setResult(null);
      setDirty(false);
      setActive(SUGGESTED[0]);
      setMessage("Calibration cleared.");
      onSaved(null);
    } catch (e) {
      setError(e);
    } finally {
      setBusy(null);
    }
  }

  const errFor = (id: string) => {
    if (!result) return null;
    const i = result.point_ids.indexOf(id);
    return i < 0 ? null : { m: result.errors_m[i], px: result.errors_px[i], inlier: result.inliers[i] };
  };
  const r = 0.008 * W;

  return (
    <div className="grid gap-5 xl:grid-cols-[minmax(0,1fr)_340px]">
      <Card
        title="Court calibration"
        subtitle="Choose a landmark on the court diagram, then click it on the frame. Drag points to adjust; right-click to remove."
        actions={
          <>
            <Button onClick={autoDetect} loading={busy === "auto"}>
              <Wand2 className="h-4 w-4" /> Auto-detect court
            </Button>
            <Button variant="ghost" onClick={reset} loading={busy === "reset"}>
              <RotateCcw className="h-4 w-4" /> Reset
            </Button>
            <Button variant="primary" onClick={save} loading={busy === "save"} disabled={!result || !!fitError || (!dirty && !!existing)}>
              <Save className="h-4 w-4" /> Save calibration
            </Button>
          </>
        }
      >
        <div className="relative overflow-hidden rounded-xl border border-edge bg-black">
          <img src={mediaUrl(`/api/videos/${session.video.id}/frame`, { t: frameT, max_width: 1600 })} alt="Video frame" className="block w-full select-none" draggable={false} />
          <svg
            ref={svgRef}
            viewBox={`0 0 ${W} ${H}`}
            preserveAspectRatio="none"
            className="absolute inset-0 h-full w-full cursor-crosshair"
            onClick={(e) => {
              if (dragging.current) return;
              const [x, y] = toImage(e);
              place(x, y);
            }}
            onPointerMove={(e) => {
              if (!dragging.current) return;
              const [x, y] = toImage(e);
              const id = dragging.current;
              setDirty(true);
              setPoints((ps) => ps.map((p) => (p.id === id ? { ...p, x, y } : p)));
            }}
            onPointerUp={() => {
              window.setTimeout(() => (dragging.current = null), 0);
            }}
          >
            {overlay.map(({ ln, runs }) =>
              runs.map((run, i) => (
                <polyline
                  key={ln.id + i}
                  points={run.map((p) => p.join(",")).join(" ")}
                  fill="none"
                  stroke={ln.kind === "net" ? "#fbbf24" : "#22d3ee"}
                  strokeOpacity={0.8}
                  strokeWidth={W / 700}
                  strokeDasharray={ln.kind === "net" ? `${W / 150} ${W / 250}` : undefined}
                />
              )),
            )}
            {points.map((p) => {
              const e = errFor(p.id);
              const color = e && !e.inlier ? "#f87171" : p.id === active ? "#22d3ee" : "#a3e635";
              return (
                <g
                  key={p.id}
                  onPointerDown={(ev) => {
                    ev.stopPropagation();
                    (ev.target as Element).setPointerCapture?.(ev.pointerId);
                    dragging.current = p.id;
                    setActive(p.id);
                  }}
                  onContextMenu={(ev) => {
                    ev.preventDefault();
                    setDirty(true);
                    setPoints((ps) => ps.filter((q) => q.id !== p.id));
                  }}
                  className="cursor-grab"
                >
                  <circle cx={p.x} cy={p.y} r={r * 1.8} fill="transparent" />
                  <circle cx={p.x} cy={p.y} r={r} fill="none" stroke={color} strokeWidth={W / 600} />
                  <line x1={p.x - r * 1.6} x2={p.x + r * 1.6} y1={p.y} y2={p.y} stroke={color} strokeWidth={W / 1200} />
                  <line y1={p.y - r * 1.6} y2={p.y + r * 1.6} x1={p.x} x2={p.x} stroke={color} strokeWidth={W / 1200} />
                  <title>{labelOf(p.id)}</title>
                </g>
              );
            })}
          </svg>
        </div>
        <div className="mt-4 flex items-center gap-3 text-xs text-muted">
          <span className="w-24">Frame {fmtTime(t)}</span>
          <input
            type="range"
            min={0}
            max={Math.max(0, session.video.duration_s - 0.1)}
            step={0.1}
            value={t}
            onChange={(e) => setT(Number(e.target.value))}
            onPointerUp={() => setFrameT(t)}
            onKeyUp={() => setFrameT(t)}
            className="flex-1"
          />
          <span>Pick a frame where the lines are least occluded</span>
        </div>
        {message && <div className="mt-3"><Notice tone="info">{message}</Notice></div>}
        {error ? <div className="mt-3"><ErrorBox error={error} /></div> : null}
      </Card>

      <div className="space-y-5">
        <Card title="Landmark picker" subtitle="Next click places:">
          <div className="mb-3 flex items-center gap-2 rounded-lg border border-cyan-400/30 bg-cyan-400/5 px-3 py-2 text-xs text-accent">
            <Crosshair className="h-4 w-4 shrink-0" /> {labelOf(active)}
          </div>
          {court && (
            <CourtSvg court={court} className="mx-auto w-full max-w-[220px]">
              {court.keypoints.map((k) => {
                const [x, y] = toSvg(court, k.x, k.y);
                const used = points.some((p) => p.id === k.id);
                const isActive = k.id === active;
                return (
                  <g key={k.id} onClick={() => setActive(k.id)} className="cursor-pointer">
                    <circle cx={x} cy={y} r={0.32} fill="transparent" />
                    <circle
                      cx={x}
                      cy={y}
                      r={isActive ? 0.22 : 0.14}
                      fill={used ? "#a3e635" : isActive ? "#22d3ee" : "#334866"}
                      stroke={isActive ? "#22d3ee" : "none"}
                      strokeWidth={0.06}
                    />
                    <title>{k.label}</title>
                  </g>
                );
              })}
            </CourtSvg>
          )}
          <p className="mt-2 text-[11px] text-muted">Near = closest to the camera. Spread points over the visible court; avoid using only one line.</p>
        </Card>

        <Card
          title={
            <span className="flex items-center gap-2">
              Validation <InfoTip text="Reprojection error: distance between each clicked landmark (mapped through the fitted homography) and its true court position. With 6+ points, RANSAC rejects mis-clicks (red)." />
            </span>
          }
          actions={result ? <Badge tone={QUALITY_TONE[result.quality]}>{result.quality}</Badge> : undefined}
        >
          {fitError ? (
            <div className="text-xs text-warn">{fitError}</div>
          ) : result ? (
            <div className="space-y-3 text-xs">
              <div className="grid grid-cols-2 gap-2 tabular">
                <div className="rounded-lg bg-navy-850 p-2">
                  <div className="text-muted">Mean error</div>
                  <div className="text-base font-semibold text-white">{fmtNum(result.mean_error_m * 100, 1)} cm</div>
                </div>
                <div className="rounded-lg bg-navy-850 p-2">
                  <div className="text-muted">Mean error (image)</div>
                  <div className="text-base font-semibold text-white">{fmtNum(result.mean_error_px, 1)} px</div>
                </div>
              </div>
              <table className="w-full">
                <thead className="text-muted">
                  <tr>
                    <th className="py-1 text-left font-medium">Landmark</th>
                    <th className="text-right font-medium">m</th>
                    <th className="text-right font-medium">px</th>
                    <th />
                  </tr>
                </thead>
                <tbody>
                  {points.map((p) => {
                    const e = errFor(p.id);
                    return (
                      <tr key={p.id} className={cx("border-t border-edge/60", e && !e.inlier && "text-bad")}>
                        <td className="max-w-[150px] truncate py-1" title={labelOf(p.id)}>
                          {labelOf(p.id)}
                        </td>
                        <td className="text-right tabular">{e ? fmtNum(e.m, 3) : "—"}</td>
                        <td className="text-right tabular">{e ? fmtNum(e.px, 1) : "—"}</td>
                        <td className="text-right">
                          <button
                            onClick={() => {
                              setDirty(true);
                              setPoints((ps) => ps.filter((q) => q.id !== p.id));
                            }}
                            className="text-muted hover:text-bad"
                            aria-label="Remove point"
                          >
                            <Trash2 className="h-3.5 w-3.5" />
                          </button>
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
              {result.warnings.map((w) => (
                <div key={w} className="text-warn">
                  • {w}
                </div>
              ))}
              {result.saved && !dirty && <div className="text-good">✓ Saved and used for analysis</div>}
            </div>
          ) : (
            <div className="text-xs text-muted">Place at least 4 landmarks, or use auto-detect.</div>
          )}
        </Card>
      </div>
    </div>
  );
}
