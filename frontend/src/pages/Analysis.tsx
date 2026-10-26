import { ArrowLeft, Download, Settings2 } from "lucide-react";
import { useCallback, useMemo, useRef, useState } from "react";
import { Link, useParams } from "react-router-dom";
import { ShareBars, SpeedChart, SpeedHistogram, ZoneGrid } from "../charts/Charts";
import { AnnotatedClipButton } from "../components/AnnotatedClipButton";
import { AnnotationEditor } from "../components/AnnotationEditor";
import { EvaluationPanel } from "../components/EvaluationPanel";
import { ExportPanel } from "../components/ExportPanel";
import { HeatmapPanel } from "../components/HeatmapPanel";
import { JobProgress } from "../components/JobProgress";
import { PosePanel } from "../components/PosePanel";
import { IdentityPanel, QualityPanel } from "../components/QualityPanel";
import { StrokePanel } from "../components/StrokePanel";
import { Badge, Card, cx, ErrorBox, ExperimentalBadge, InfoTip, Notice, Skeleton, Spinner, Stat, SyntheticBanner, Toggle } from "../components/ui";
import { VideoAnalyzer } from "../components/VideoAnalyzer";
import { useAsync } from "../hooks/useAsync";
import { useCourt } from "../hooks/useStatic";
import { clipRange } from "../lib/clip";
import { DEF } from "../lib/definitions";
import { fmtNum, fmtPct, fmtTime, titleCase } from "../lib/format";
import { api, ApiError } from "../services/api";
import type { Metrics, MetricsResponse, Stroke } from "../types/api";

type Tab = "overview" | "pose" | "strokes" | "quality" | "evaluation" | "export";

function FootworkPanel({ m }: { m: Metrics }) {
  const [showComposite, setShowComposite] = useState(true);
  const fw = m.footwork;
  const rec = m.recovery;
  const comp = m.composite;
  if (!fw || !rec) return null;
  const rows: [string, string, string][] = [
    ["Movements away from base", String(rec.excursions), DEF.recovery],
    ["Median recovery time", fmtNum(rec.recovery_time_median_s, 2, "s"), DEF.recovery],
    ["Time > 1.5 m from base", fmtPct(rec.far_share), DEF.recovery],
    ["Mean distance from base", fmtNum(rec.mean_distance_from_base_m, 2, "m"), DEF.recovery],
    ["Path directness", fmtNum(fw.path_directness_mean, 2), DEF.directness],
    ["Extra distance", fmtNum(fw.extra_distance_pct_mean, 1, "%"), DEF.extraDistance],
    ["Acceleration p95", fmtNum(fw.peak_acceleration_p95_mps2, 1, "m/s²"), DEF.accel],
    ["Deceleration p95", fmtNum(fw.peak_deceleration_p95_mps2, 1, "m/s²"), DEF.decel],
    ["Smoothness (LDLJ)", fmtNum(fw.smoothness_ldlj_median, 1), DEF.smoothness],
    ["Ready-position spread", fmtNum(fw.positioning_consistency_m, 2, "m"), DEF.consistency],
  ];
  return (
    <Card
      title={
        <span className="flex items-center gap-2">
          Recovery & footwork <Badge tone="neutral">movement proxies</Badge>
        </span>
      }
      subtitle={`Base position: ${fmtNum(rec.base_position_half.lateral_m, 2)} m from the left singles line, ${fmtNum(rec.base_position_half.depth_from_net_m, 1)} m from the net`}
    >
      <div className="space-y-1">
        {rows.map(([k, v, hint]) => (
          <div key={k} className="flex items-center justify-between border-b border-edge/50 py-1.5 text-sm last:border-0">
            <span className="flex items-center gap-1.5 text-muted">
              {k} <InfoTip text={hint} />
            </span>
            <span className="tabular text-soft">{v}</span>
          </div>
        ))}
      </div>
      {comp && (
        <div className="mt-4 rounded-xl border border-amber-400/30 bg-amber-400/5 p-4">
          <div className="flex items-center justify-between gap-2">
            <div className="flex items-center gap-2 text-sm font-semibold text-white">
              Footwork index <ExperimentalBadge />
            </div>
            <Toggle checked={showComposite} onChange={setShowComposite} label="Show" />
          </div>
          {showComposite && (
            <div className="mt-3 space-y-2 text-xs">
              <div className="text-3xl font-bold tabular text-warn">
                {fmtNum(comp.score, 0)}
                <span className="text-sm text-muted"> / 100</span>
              </div>
              <div className="grid grid-cols-2 gap-1">
                {Object.entries(comp.weights).map(([k, w]) => (
                  <div key={k} className="flex justify-between rounded bg-navy-850 px-2 py-1">
                    <span className="text-muted">{titleCase(k)}</span>
                    <span className="tabular text-soft">
                      {fmtNum(comp.components[k], 2)} × {w}
                    </span>
                  </div>
                ))}
              </div>
              <p className="leading-relaxed text-muted">{comp.formula}</p>
              <p className="leading-relaxed text-warn/90">{comp.caveat}</p>
            </div>
          )}
        </div>
      )}
    </Card>
  );
}

function KeyStats({ m }: { m: Metrics }) {
  return (
    <div className="grid grid-cols-2 gap-3">
      <Stat label="Distance" value={fmtNum(m.distance?.total_m, 1)} unit="m" hint={DEF.distance} sub={`${fmtNum(m.distance?.interpolated_m, 1)} m across short gaps`} />
      <Stat label="Distance / min" value={fmtNum(m.distance?.per_minute_m, 1)} unit="m" hint={DEF.distancePerMin} />
      <Stat label="Mean speed" value={fmtNum(m.speed?.mean_mps, 2)} unit="m/s" hint={DEF.meanSpeed} sub={`${fmtNum(m.speed?.mean_while_moving_mps, 2)} m/s while moving`} />
      <Stat label="p95 / max speed" value={fmtNum(m.speed?.p95_mps, 2)} unit="m/s" hint={`${DEF.p95Speed} ${DEF.maxSpeed}`} sub={`max ${fmtNum(m.speed?.max_filtered_mps, 2)} m/s (filtered)`} tone="good" />
      <Stat label="Moving" value={fmtPct(m.speed?.moving_share)} hint={DEF.movingShare} sub={`${fmtNum(m.speed?.time_moving_s, 1)} s of ${fmtNum(m.observed_s, 1)} s`} />
      <Stat label="Direction changes" value={m.direction_changes ?? "—"} hint={DEF.directionChanges} sub={`${fmtNum(m.direction_changes_per_minute, 1)} per min`} />
      <Stat label="Zones visited" value={fmtNum(m.zones?.visited_pct, 0)} unit="%" hint={DEF.zonesVisited} sub={`${m.zones?.frequented.length ?? 0} zones ≥ 5% of time`} />
      <Stat label="Recovery (median)" value={fmtNum(m.recovery?.recovery_time_median_s, 2)} unit="s" hint={DEF.recovery} sub="movement proxy" />
      <Stat label="Area for 90% of time" value={fmtNum(m.heatmap_area_90_m2, 1)} unit="m²" hint={DEF.area90} />
      <Stat label="Observed" value={fmtNum(m.coverage_pct, 0)} unit="%" hint={DEF.coverage} sub={`${fmtNum(m.observed_s, 1)} s with a court position`} />
    </div>
  );
}

export function AnalysisPage() {
  const { id = "" } = useParams();
  const court = useCourt();
  const sess = useAsync(() => api.session(id), [id]);
  const s = sess.data;
  const ready = s?.status === "completed";
  const metrics = useAsync<MetricsResponse>(() => api.metrics(id), [id, s?.latest_job?.id], !!s);
  const traj = useAsync(() => api.trajectory(id, { max_points: 20000 }), [id, s?.latest_job?.id], ready);
  const events = useAsync(() => api.events(id), [id, s?.latest_job?.id], ready);
  const strokes = useAsync<Stroke[]>(() => api.strokes(id), [id, s?.latest_job?.id], ready);
  const [tab, setTab] = useState<Tab>("overview");
  const [interval, setInterval] = useState<[number, number] | null>(null);
  const [coarseTime, setCoarseTime] = useState(0);
  const timeRef = useRef(0);
  const seekRef = useRef<((t: number) => void) | null>(null);

  const onTime = useCallback((t: number) => {
    timeRef.current = t;
    setCoarseTime((prev) => (Math.abs(prev - t) > 0.25 ? t : prev));
  }, []);
  const seek = useCallback((t: number) => {
    seekRef.current?.(t);
    setTab((x) => (x === "overview" || x === "strokes" || x === "quality" || x === "pose" ? x : "overview"));
  }, []);
  const timelineEvents = useMemo(() => (events.data ?? []).filter((e) => e.type !== "stroke"), [events.data]);

  if (sess.error) return <ErrorBox error={sess.error} />;
  if (!s) return <Spinner label="Loading analysis…" />;
  const synthetic = s.data_source === "synthetic" || !!s.summary?.synthetic;

  if (!ready) {
    return (
      <div className="mx-auto max-w-3xl space-y-4">
        <h1 className="text-2xl font-bold text-white">{s.name}</h1>
        {synthetic && <SyntheticBanner />}
        <Card title="Analysis not available yet">
          {s.latest_job && !["COMPLETED", "FAILED", "CANCELLED"].includes(s.latest_job.status) ? (
            <JobProgress jobId={s.latest_job.id} onDone={() => sess.reload()} />
          ) : (
            <div className="space-y-3 text-sm text-muted">
              {s.latest_job?.error && <ErrorBox error={s.latest_job.error} />}
              <p>This session has no completed analysis.</p>
              <Link to={`/sessions/${id}/setup`} className="text-accent hover:underline">
                Open the setup to calibrate, select the player and start the analysis →
              </Link>
            </div>
          )}
        </Card>
      </div>
    );
  }

  const m = metrics.data?.metrics;
  const calibrated = m ? m.calibrated !== false : true;
  const points = traj.data?.points ?? [];
  const tabs: { id: Tab; label: string }[] = [
    { id: "overview", label: "Overview" },
    { id: "pose", label: "Pose" },
    { id: "strokes", label: `Strokes (${strokes.data?.length ?? 0})` },
    { id: "quality", label: "Quality & identity" },
    { id: "evaluation", label: "Evaluation" },
    { id: "export", label: "Export" },
  ];
  const clip = () => {
    const d = s.video.duration_s;
    return interval ? clipRange(interval[0], interval[1], d) : clipRange(timeRef.current, timeRef.current + 20, d);
  };

  return (
    <div className="space-y-5">
      <div className="flex flex-wrap items-end justify-between gap-3">
        <div>
          <Link to="/" className="mb-1 inline-flex items-center gap-1 text-xs text-muted hover:text-accent">
            <ArrowLeft className="h-3 w-3" /> Dashboard
          </Link>
          <h1 className="text-2xl font-bold tracking-tight text-white">{s.name}</h1>
          <div className="mt-1 flex flex-wrap items-center gap-2 text-xs text-muted">
            <span>
              {s.video.width}×{s.video.height} · {fmtNum(s.video.fps, 0)} fps · {fmtTime(s.video.duration_s, 0)}
            </span>
            {metrics.data && (
              <>
                <span>·</span>
                <span>
                  {metrics.data.analysis.performance_mode} mode, {fmtNum(metrics.data.analysis.fps, 0)} fps analysis, {metrics.data.analysis.detector} + {metrics.data.analysis.tracker}
                </span>
              </>
            )}
          </div>
        </div>
        <div className="flex gap-2">
          <Link to={`/sessions/${id}/setup`} className="inline-flex items-center gap-1.5 rounded-lg border border-edge px-3 py-2 text-sm text-soft hover:border-accent hover:text-accent">
            <Settings2 className="h-4 w-4" /> Setup & re-run
          </Link>
          <button onClick={() => setTab("export")} className="inline-flex items-center gap-1.5 rounded-lg bg-accent px-3 py-2 text-sm font-medium text-navy-900 hover:bg-cyan-300">
            <Download className="h-4 w-4" /> Export
          </button>
        </div>
      </div>
      {synthetic && <SyntheticBanner />}
      {metrics.error ? (
        <ErrorBox error={metrics.error instanceof ApiError && metrics.error.status === 404 ? "No results found for this session — re-run the analysis." : metrics.error} />
      ) : null}
      {m && !calibrated && <Notice>{m.reason}</Notice>}

      <div className="grid gap-5 xl:grid-cols-3">
        <div className="xl:col-span-2">
          <Card className="h-full">
            <VideoAnalyzer
              session={s}
              court={court}
              events={timelineEvents}
              strokes={strokes.data ?? []}
              interval={interval}
              onInterval={setInterval}
              seekRef={seekRef}
              onTime={onTime}
            />
            <div className="mt-3 flex flex-wrap items-center gap-3 border-t border-edge pt-3 text-xs text-muted">
              <AnnotatedClipButton sessionId={id} range={clip} label={interval ? "Download annotated clip (selection)" : "Download annotated clip (next 20 s)"} />
              <span>Overlays burned in; at most 60 s. Uses the timeline selection, or 20 s from the current time.</span>
            </div>
          </Card>
        </div>
        <div>{m && calibrated ? <KeyStats m={m} /> : metrics.loading ? <Skeleton className="h-full min-h-80" /> : null}</div>
      </div>

      <div className="flex gap-1 overflow-x-auto border-b border-edge">
        {tabs.map((t) => (
          <button
            key={t.id}
            onClick={() => setTab(t.id)}
            className={cx(
              "whitespace-nowrap border-b-2 px-4 py-2.5 text-sm font-medium transition",
              tab === t.id ? "border-accent text-accent" : "border-transparent text-muted hover:text-soft",
            )}
          >
            {t.label}
          </button>
        ))}
      </div>

      {tab === "overview" && m && (
        <div className="grid gap-5 xl:grid-cols-3 fade-in">
          <div className="xl:col-span-2">
            <HeatmapPanel sessionId={id} points={points} interval={interval} synthetic={synthetic} calibrated={calibrated} reason={m.reason} />
          </div>
          <div className="space-y-5">
            {calibrated && m.zones && (
              <Card title={<span className="flex items-center gap-2">Zone occupancy <InfoTip text={DEF.zonesVisited} /></span>} subtitle="Singles half-court, from the player's point of view">
                <ZoneGrid zones={m.zones} />
                <div className="mt-4 grid grid-cols-2 gap-4">
                  <ShareBars shares={m.zones.depth_share} />
                  <ShareBars shares={m.zones.lateral_share} labels={{ center: "Centre" }} />
                </div>
                <div className="mt-3 text-xs text-muted">Outside the singles lines: {fmtNum(m.zones.outside_singles_s, 1)} s</div>
              </Card>
            )}
          </div>
          {calibrated && (
            <>
              <div className="xl:col-span-2">
                <Card title="Speed over time" subtitle="Filtered speed (m/s). Dashed line: 4 m/s high-speed threshold. Click to seek.">
                  {traj.loading ? <Skeleton className="h-56" /> : <SpeedChart points={points} time={coarseTime} interval={interval} onSeek={seek} />}
                  {m.speed && (
                    <div className="mt-4">
                      <div className="mb-1 text-xs font-medium text-muted">Time spent per speed band (s)</div>
                      <SpeedHistogram hist={m.speed.histogram} />
                    </div>
                  )}
                </Card>
              </div>
              <FootworkPanel m={m} />
            </>
          )}
        </div>
      )}
      {tab === "pose" && <PosePanel sessionId={id} onSeek={seek} />}
      {tab === "strokes" && (
        <StrokePanel
          sessionId={id}
          strokes={strokes.data ?? []}
          setStrokes={(fn) => strokes.setData((prev) => fn(prev ?? []))}
          info={m?.strokes}
          currentTime={() => timeRef.current}
          onSeek={seek}
        />
      )}
      {tab === "quality" && metrics.data && (
        <div className="grid gap-5 xl:grid-cols-2">
          <QualityPanel m={metrics.data} />
          <IdentityPanel session={s} currentTime={() => timeRef.current} onRerunDone={() => sess.reload()} />
        </div>
      )}
      {tab === "evaluation" && (
        <div className="space-y-5">
          <EvaluationPanel sessionId={id} />
          <AnnotationEditor sessionId={id} videoId={s.video_id} width={s.video.width} height={s.video.height} duration={s.video.duration_s} />
        </div>
      )}
      {tab === "export" && (
        <div className="space-y-5">
          <ExportPanel sessionId={id} synthetic={synthetic} />
          <Card title="Annotated video clip" subtitle="Rendered on the server with boxes, skeleton, court lines and trail burned in (≤ 60 s).">
            <AnnotatedClipButton sessionId={id} range={clip} />
            <p className="mt-2 text-xs text-muted">
              Range: {interval ? `timeline selection ${fmtTime(clip()[0])}–${fmtTime(clip()[1])}` : `20 s from the current playback time (${fmtTime(timeRef.current)})`}
            </p>
          </Card>
        </div>
      )}
    </div>
  );
}
