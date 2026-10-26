import { Activity, ArrowRight, FlaskConical, Gauge, Route, Timer, Trash2, Upload, Video } from "lucide-react";
import { useState } from "react";
import { Link, useNavigate } from "react-router-dom";
import { JobProgress } from "../components/JobProgress";
import { Badge, Button, Card, EmptyState, ErrorBox, Notice, Skeleton, Stat, SyntheticBadge } from "../components/ui";
import { useAsync } from "../hooks/useAsync";
import { DEF } from "../lib/definitions";
import { fmtDuration, fmtNum, relativeDate } from "../lib/format";
import { api, mediaUrl } from "../services/api";
import type { Session } from "../types/api";

const STATUS_TONE: Record<Session["status"], "neutral" | "accent" | "good" | "warn" | "bad"> = {
  draft: "neutral",
  queued: "accent",
  processing: "accent",
  completed: "good",
  failed: "bad",
  cancelled: "warn",
};

function sessionLink(s: Session): string {
  return s.status === "completed" ? `/sessions/${s.id}` : `/sessions/${s.id}/setup`;
}

function SessionCard({ s, onDelete, onDone }: { s: Session; onDelete: () => void; onDone: () => void }) {
  const synthetic = s.data_source === "synthetic" || s.summary?.synthetic;
  const running = s.status === "queued" || s.status === "processing";
  return (
    <div className="group fade-in overflow-hidden rounded-2xl border border-edge bg-navy-800/80 transition hover:-translate-y-0.5 hover:border-cyan-400/40 hover:shadow-[0_12px_40px_-12px_rgba(34,211,238,0.25)]">
      <Link to={sessionLink(s)} className="relative block aspect-video overflow-hidden bg-ink">
        <img src={mediaUrl(s.video.thumbnail_url)} alt="" className="h-full w-full object-cover opacity-80 transition group-hover:scale-[1.02] group-hover:opacity-100" loading="lazy" />
        <div className="absolute left-3 top-3 flex gap-1.5">
          <Badge tone={STATUS_TONE[s.status]}>{s.status}</Badge>
          {synthetic && <SyntheticBadge />}
        </div>
        <span className="absolute bottom-2 right-2 rounded bg-black/70 px-1.5 py-0.5 text-[11px] tabular text-white">{fmtDuration(s.video.duration_s)}</span>
      </Link>
      <div className="space-y-3 p-4">
        <div className="flex items-start justify-between gap-2">
          <div className="min-w-0">
            <Link to={sessionLink(s)} className="block truncate font-semibold text-white hover:text-accent">
              {s.name}
            </Link>
            <div className="text-xs text-muted">
              {relativeDate(s.created_at)} · {s.video.width}×{s.video.height} · {fmtNum(s.video.fps, 0)} fps
            </div>
          </div>
          <button
            onClick={onDelete}
            className="rounded-md p-1.5 text-muted opacity-0 transition hover:bg-red-500/10 hover:text-bad group-hover:opacity-100"
            aria-label="Delete session"
            title="Delete session"
          >
            <Trash2 className="h-4 w-4" />
          </button>
        </div>
        {running && s.latest_job ? (
          <JobProgress jobId={s.latest_job.id} compact onDone={onDone} />
        ) : s.status === "completed" && s.summary?.calibrated !== false ? (
          <div className="grid grid-cols-3 gap-2 text-center">
            <div>
              <div className="text-[10px] uppercase tracking-wider text-muted">Distance</div>
              <div className="font-semibold tabular text-white">{fmtNum(s.summary.distance_m, 0)} m</div>
            </div>
            <div>
              <div className="text-[10px] uppercase tracking-wider text-muted">p95 speed</div>
              <div className="font-semibold tabular text-white">{fmtNum(s.summary.p95_speed_mps, 1)}</div>
            </div>
            <div>
              <div className="text-[10px] uppercase tracking-wider text-muted">Tracked</div>
              <div className="font-semibold tabular text-good">{fmtNum(s.summary.target_coverage_pct, 0)}%</div>
            </div>
          </div>
        ) : s.status === "completed" ? (
          <div className="text-xs text-warn">Analysed without court calibration — movement metrics unavailable.</div>
        ) : (
          <Link to={`/sessions/${s.id}/setup`} className="inline-flex items-center gap-1 text-xs font-medium text-accent hover:underline">
            Continue setup <ArrowRight className="h-3 w-3" />
          </Link>
        )}
      </div>
    </div>
  );
}

export function Dashboard() {
  const sessions = useAsync(() => api.sessions(), []);
  const navigate = useNavigate();
  const [demoBusy, setDemoBusy] = useState<number | null>(null);
  const [error, setError] = useState<unknown>(null);
  const list = sessions.data ?? [];
  const done = list.filter((s) => s.status === "completed" && s.summary?.calibrated !== false);
  const real = done.filter((s) => s.data_source !== "synthetic");
  const basis = real.length ? real : done;
  const totalDistance = basis.reduce((a, s) => a + (s.summary.distance_m ?? 0), 0);
  const totalObserved = basis.reduce((a, s) => a + (s.summary.observed_s ?? 0), 0);
  const bestP95 = Math.max(0, ...basis.map((s) => s.summary.p95_speed_mps ?? 0));
  const onlySynthetic = !real.length && done.length > 0;

  async function startDemo(variant: 1 | 2) {
    setDemoBusy(variant);
    setError(null);
    try {
      const r = await api.demo(variant);
      navigate(`/sessions/${r.session.id}/setup?job=${r.job.id}`);
    } catch (e) {
      setError(e);
    } finally {
      setDemoBusy(null);
    }
  }

  return (
    <div className="space-y-8">
      <div className="flex flex-wrap items-end justify-between gap-4">
        <div>
          <h1 className="text-2xl font-bold tracking-tight text-white md:text-3xl">Performance dashboard</h1>
          <p className="mt-1 max-w-2xl text-sm text-muted">
            Court-calibrated movement analysis from a single fixed camera: positions in metres, occupancy heatmaps, speed, coverage and
            recovery behaviour.
          </p>
        </div>
        <div className="flex flex-wrap gap-2">
          <Button variant="subtle" loading={demoBusy === 1} onClick={() => startDemo(1)} title="Render and analyse a synthetic rally">
            <FlaskConical className="h-4 w-4 text-synthetic" /> Try demo A
          </Button>
          <Button variant="subtle" loading={demoBusy === 2} onClick={() => startDemo(2)} title="A faster synthetic rally, useful for Compare">
            <FlaskConical className="h-4 w-4 text-synthetic" /> Try demo B
          </Button>
          <Button variant="primary" onClick={() => navigate("/new")}>
            <Upload className="h-4 w-4" /> Upload video
          </Button>
        </div>
      </div>
      {error ? <ErrorBox error={error} /> : null}
      {demoBusy && (
        <Notice tone="info">
          Rendering a synthetic 45-second rally video on the server (usually 20–60 s). The analysis starts automatically and you'll be taken
          to its progress page.
        </Notice>
      )}

      <div className="grid grid-cols-2 gap-4 lg:grid-cols-4">
        <Stat label="Sessions" value={list.length} sub={`${done.length} analysed`} />
        <Stat label="Distance covered" value={fmtNum(totalDistance, 0)} unit="m" hint={DEF.distance} sub={onlySynthetic ? "synthetic demos only" : "across analysed sessions"} />
        <Stat label="Observed time" value={fmtDuration(totalObserved)} sub="player tracked on court" />
        <Stat label="Best p95 speed" value={fmtNum(bestP95 || null, 2)} unit="m/s" hint={DEF.p95Speed} tone="good" sub={onlySynthetic ? "synthetic demos only" : undefined} />
      </div>

      <Card title="Sessions" subtitle="Most recent first" actions={<Button variant="ghost" onClick={sessions.reload}>Refresh</Button>}>
        {sessions.loading && !sessions.data ? (
          <div className="grid gap-4 sm:grid-cols-2 xl:grid-cols-3">
            {[0, 1, 2].map((i) => (
              <Skeleton key={i} className="h-64" />
            ))}
          </div>
        ) : sessions.error ? (
          <ErrorBox error={sessions.error} action={<Button onClick={sessions.reload}>Retry</Button>} />
        ) : list.length === 0 ? (
          <EmptyState icon={<Video className="h-10 w-10" />} title="No sessions yet">
            Upload a match or practice video filmed from behind the baseline, or try a clearly-labelled synthetic demo to explore the
            dashboard.
            <div className="mt-4 flex justify-center gap-2">
              <Button variant="primary" onClick={() => navigate("/new")}>
                <Upload className="h-4 w-4" /> Upload video
              </Button>
              <Button loading={demoBusy === 1} onClick={() => startDemo(1)}>
                <FlaskConical className="h-4 w-4" /> Try demo
              </Button>
            </div>
          </EmptyState>
        ) : (
          <div className="grid gap-5 sm:grid-cols-2 xl:grid-cols-3">
            {list.map((s) => (
              <SessionCard
                key={s.id}
                s={s}
                onDone={sessions.reload}
                onDelete={async () => {
                  if (!window.confirm(`Delete "${s.name}" and all its files? This cannot be undone.`)) return;
                  try {
                    await api.deleteSession(s.id);
                    sessions.reload();
                  } catch (e) {
                    setError(e);
                  }
                }}
              />
            ))}
          </div>
        )}
      </Card>

      <div className="grid gap-4 md:grid-cols-3">
        {[
          { icon: <Route className="h-5 w-5" />, title: "1 · Calibrate the court", body: "Click four or more court landmarks (or auto-detect) so pixels map to metres." },
          { icon: <Activity className="h-5 w-5" />, title: "2 · Pick the player", body: "Choose the player to follow; tracking links them through occlusions." },
          { icon: <Gauge className="h-5 w-5" />, title: "3 · Review the analysis", body: "Heatmaps, speed, coverage and recovery proxies with data-quality indicators." },
        ].map((x) => (
          <div key={x.title} className="flex gap-3 rounded-xl border border-edge bg-navy-850/60 p-4">
            <div className="text-accent">{x.icon}</div>
            <div>
              <div className="text-sm font-semibold text-white">{x.title}</div>
              <div className="mt-0.5 text-xs text-muted">{x.body}</div>
            </div>
          </div>
        ))}
      </div>
      <p className="flex items-center gap-1.5 text-xs text-muted">
        <Timer className="h-3.5 w-3.5" /> Processing runs on the server; CPU works everywhere, a CUDA GPU is used automatically when present.
      </p>
    </div>
  );
}
