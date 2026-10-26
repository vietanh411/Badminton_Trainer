import { Download, Flame } from "lucide-react";
import { useMemo, useState } from "react";
import { useAsync } from "../hooks/useAsync";
import { useCourt } from "../hooks/useStatic";
import { cssGradient } from "../lib/colormap";
import { DEF } from "../lib/definitions";
import { fmtNum, fmtTime } from "../lib/format";
import { api, mediaUrl } from "../services/api";
import type { TrajectoryPoint } from "../types/api";
import { CourtSvg, HeatmapLayer, toSvg } from "./CourtSvg";
import { Card, ErrorBox, InfoTip, Notice, Segmented, Skeleton, SyntheticBadge, Toggle } from "./ui";

type Preset = "full" | "first_half" | "second_half" | "interval";
type Zone = "all" | "front" | "mid" | "rear";
type Norm = "share" | "seconds" | "density";

export function HeatmapPanel({
  sessionId,
  points,
  interval,
  synthetic,
  calibrated,
  reason,
}: {
  sessionId: string;
  points: TrajectoryPoint[];
  interval: [number, number] | null;
  synthetic: boolean;
  calibrated: boolean;
  reason?: string;
}) {
  const court = useCourt();
  const [preset, setPreset] = useState<Preset>("full");
  const [zone, setZone] = useState<Zone>("all");
  const [norm, setNorm] = useState<Norm>("share");
  const [sigma, setSigma] = useState(0.3);
  const [opacity, setOpacity] = useState(0.9);
  const [showPoints, setShowPoints] = useState(false);
  const [showTrail, setShowTrail] = useState(false);
  const [zones, setZones] = useState(false);

  const effectivePreset = preset === "interval" && !interval ? "full" : preset;
  const params = useMemo(
    () => ({
      preset: effectivePreset === "first_half" || effectivePreset === "second_half" ? effectivePreset : undefined,
      start: effectivePreset === "interval" && interval ? interval[0] : undefined,
      end: effectivePreset === "interval" && interval ? interval[1] : undefined,
      zone: zone === "all" ? undefined : zone,
      sigma_m: sigma,
      normalize: norm,
    }),
    [effectivePreset, interval, zone, sigma, norm],
  );
  const hm = useAsync(() => api.heatmap(sessionId, params), [sessionId, JSON.stringify(params)], calibrated);

  const visiblePts = useMemo(() => {
    let p = points.filter((q) => q.court_x !== null && q.court_y !== null);
    if (effectivePreset === "interval" && interval) p = p.filter((q) => q.t >= interval[0] && q.t <= interval[1]);
    if (zone !== "all") p = p.filter((q) => q.zone?.startsWith(zone));
    if (effectivePreset === "first_half" || effectivePreset === "second_half") {
      const t0 = points[0]?.t ?? 0;
      const t1 = points[points.length - 1]?.t ?? 0;
      const mid = t0 + (t1 - t0) / 2;
      p = p.filter((q) => (effectivePreset === "first_half" ? q.t <= mid : q.t >= mid));
    }
    return p;
  }, [points, effectivePreset, interval, zone]);

  const exportUrl = (format: "png" | "svg") =>
    mediaUrl(`/api/sessions/${sessionId}/heatmap`, {
      ...params,
      format,
      points: showPoints,
      trajectory: showTrail,
      opacity,
    });

  if (!calibrated)
    return (
      <Card title="Movement heatmap">
        <Notice>
          Court heatmaps are disabled for this session. {reason ?? "A valid court calibration is required to map the player's feet to metres."}
        </Notice>
      </Card>
    );

  const unitsLabel = norm === "share" ? "% of observed time per cell" : norm === "seconds" ? "seconds per cell" : "seconds per m²";
  const max = hm.data?.max ?? 0;
  const legendMax = norm === "share" ? max * 100 : max;

  return (
    <Card
      title={
        <span className="flex items-center gap-2">
          <Flame className="h-4 w-4 text-accent" /> Movement heatmap <InfoTip text={DEF.heatmap} />
        </span>
      }
      subtitle="Time-weighted court occupancy, top-down (far baseline at the top)"
      actions={
        <>
          {synthetic && <SyntheticBadge />}
          <a className="inline-flex items-center gap-1 rounded-lg border border-edge px-2.5 py-1.5 text-xs text-soft hover:border-accent hover:text-accent" href={exportUrl("png")}>
            <Download className="h-3.5 w-3.5" /> PNG
          </a>
          <a className="inline-flex items-center gap-1 rounded-lg border border-edge px-2.5 py-1.5 text-xs text-soft hover:border-accent hover:text-accent" href={exportUrl("svg")}>
            <Download className="h-3.5 w-3.5" /> SVG
          </a>
        </>
      }
    >
      <div className="grid gap-5 lg:grid-cols-[minmax(0,1fr)_260px]">
        <div className="relative mx-auto w-full max-w-[380px]">
          {!court || (hm.loading && !hm.data) ? (
            <Skeleton className="aspect-[8.1/15.4] w-full" />
          ) : (
            <CourtSvg court={court} className="w-full drop-shadow-[0_10px_30px_rgba(34,211,238,0.08)]" showZones={zones}
              below={hm.data?.available ? <HeatmapLayer court={court} hm={hm.data} opacity={opacity} /> : null}
            >
              {showTrail && visiblePts.length > 1 && (
                <polyline
                  fill="none"
                  stroke="#22d3ee"
                  strokeOpacity={0.55}
                  strokeWidth={0.03}
                  points={visiblePts.map((p) => toSvg(court, p.court_x!, p.court_y!).join(",")).join(" ")}
                />
              )}
              {showPoints &&
                visiblePts.map((p, i) => {
                  const [x, y] = toSvg(court, p.court_x!, p.court_y!);
                  return <circle key={i} cx={x} cy={y} r={0.035} fill={p.interpolated ? "#fbbf24" : "#a3e635"} fillOpacity={0.7} />;
                })}
            </CourtSvg>
          )}
          {hm.loading && hm.data && <div className="absolute right-2 top-2 rounded bg-navy-900/80 px-2 py-0.5 text-[10px] text-muted">updating…</div>}
        </div>
        <div className="space-y-4 text-xs">
          {hm.error ? <ErrorBox error={hm.error} /> : null}
          <div>
            <div className="mb-1.5 font-medium text-muted">Time range</div>
            <Segmented<Preset>
              value={preset}
              onChange={setPreset}
              options={[
                { value: "full", label: "Full" },
                { value: "first_half", label: "1st half" },
                { value: "second_half", label: "2nd half" },
                { value: "interval", label: "Selection", title: "Drag on the video timeline to select an interval" },
              ]}
            />
            {preset === "interval" && (
              <div className="mt-1.5 text-muted">
                {interval ? `${fmtTime(interval[0])} – ${fmtTime(interval[1])}` : "Drag across the timeline under the video to select an interval."}
              </div>
            )}
          </div>
          <div>
            <div className="mb-1.5 font-medium text-muted">Court area</div>
            <Segmented<Zone>
              value={zone}
              onChange={setZone}
              options={[
                { value: "all", label: "All" },
                { value: "front", label: "Front" },
                { value: "mid", label: "Mid" },
                { value: "rear", label: "Rear" },
              ]}
            />
          </div>
          <div>
            <div className="mb-1.5 flex items-center gap-1 font-medium text-muted">
              Normalisation
              <InfoTip text="Share: fraction of observed time (sums to 100%, comparable across sessions). Seconds: absolute time per 10 cm cell. Density: seconds per m²." />
            </div>
            <Segmented<Norm>
              value={norm}
              onChange={setNorm}
              options={[
                { value: "share", label: "Share" },
                { value: "seconds", label: "Seconds" },
                { value: "density", label: "s/m²" },
              ]}
            />
          </div>
          <label className="block">
            <span className="mb-1 flex justify-between text-muted">
              Smoothing (Gaussian σ) <span className="tabular text-soft">{sigma.toFixed(2)} m</span>
            </span>
            <input type="range" min={0} max={1} step={0.05} value={sigma} onChange={(e) => setSigma(Number(e.target.value))} className="w-full" />
          </label>
          <label className="block">
            <span className="mb-1 flex justify-between text-muted">
              Opacity <span className="tabular text-soft">{Math.round(opacity * 100)}%</span>
            </span>
            <input type="range" min={0.2} max={1} step={0.05} value={opacity} onChange={(e) => setOpacity(Number(e.target.value))} className="w-full" />
          </label>
          <div className="flex flex-col gap-2">
            <Toggle checked={showPoints} onChange={setShowPoints} label="Raw position points" />
            <Toggle checked={showTrail} onChange={setShowTrail} label="Trajectory lines" />
            <Toggle checked={zones} onChange={setZones} label="Zone guides" />
          </div>
          <div>
            <div className="mb-1 h-2.5 rounded-full" style={{ background: cssGradient() }} />
            <div className="flex justify-between tabular text-muted">
              <span>0</span>
              <span>
                {fmtNum(legendMax, norm === "share" ? 2 : 2)} {norm === "share" ? "%" : ""}
              </span>
            </div>
            <div className="mt-0.5 text-muted">{unitsLabel}</div>
          </div>
          {hm.data?.available && (
            <div className="grid grid-cols-2 gap-2 rounded-lg border border-edge bg-navy-850 p-3 tabular">
              <div>
                <div className="text-muted">Observed</div>
                <div className="text-soft">{fmtNum(hm.data.observed_s, 1, "s")}</div>
              </div>
              <div>
                <div className="flex items-center gap-1 text-muted">
                  90% area <InfoTip text={DEF.area90} />
                </div>
                <div className="text-soft">{fmtNum(hm.data.area_90_m2, 1, "m²")}</div>
              </div>
            </div>
          )}
          {showPoints && <div className="text-muted"><span className="text-good">●</span> measured <span className="ml-2 text-warn">●</span> interpolated (short gap)</div>}
        </div>
      </div>
    </Card>
  );
}
