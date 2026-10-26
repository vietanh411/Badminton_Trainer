import { ArrowLeft } from "lucide-react";
import { useState } from "react";
import { Link, useParams } from "react-router-dom";
import { AnnotatedClipButton } from "../components/AnnotatedClipButton";
import { ExportPanel } from "../components/ExportPanel";
import { Card, ErrorBox, Field, inputClass, Notice, Spinner, Stat, SyntheticBanner } from "../components/ui";
import { useAsync } from "../hooks/useAsync";
import { clipRange } from "../lib/clip";
import { DEF } from "../lib/definitions";
import { fmtNum, fmtPct, fmtTime } from "../lib/format";
import { api } from "../services/api";

export function ExportPage() {
  const { id = "" } = useParams();
  const sess = useAsync(() => api.session(id), [id]);
  const metrics = useAsync(() => api.metrics(id), [id]);
  const [start, setStart] = useState("0");
  const [end, setEnd] = useState("20");
  if (sess.error) return <ErrorBox error={sess.error} />;
  if (!sess.data) return <Spinner />;
  const s = sess.data;
  const synthetic = s.data_source === "synthetic";
  const m = metrics.data?.metrics;
  const range = clipRange(Number(start) || 0, Number(end) || 0, s.video.duration_s);
  return (
    <div className="space-y-5">
      <div>
        <Link to={`/sessions/${id}`} className="mb-1 inline-flex items-center gap-1 text-xs text-muted hover:text-accent">
          <ArrowLeft className="h-3 w-3" /> Back to analysis
        </Link>
        <h1 className="text-2xl font-bold tracking-tight text-white">Export · {s.name}</h1>
      </div>
      {synthetic && <SyntheticBanner />}
      {metrics.error ? <ErrorBox error={metrics.error} /> : null}
      {m && m.calibrated !== false && (
        <Card title="Performance summary">
          <div className="grid grid-cols-2 gap-3 md:grid-cols-3 xl:grid-cols-6">
            <Stat label="Distance" value={fmtNum(m.distance?.total_m, 1)} unit="m" hint={DEF.distance} />
            <Stat label="Distance / min" value={fmtNum(m.distance?.per_minute_m, 1)} unit="m" hint={DEF.distancePerMin} />
            <Stat label="Mean speed" value={fmtNum(m.speed?.mean_mps, 2)} unit="m/s" hint={DEF.meanSpeed} />
            <Stat label="p95 speed" value={fmtNum(m.speed?.p95_mps, 2)} unit="m/s" hint={DEF.p95Speed} />
            <Stat label="Moving" value={fmtPct(m.speed?.moving_share)} hint={DEF.movingShare} />
            <Stat label="Zones visited" value={fmtNum(m.zones?.visited_pct, 0)} unit="%" hint={DEF.zonesVisited} />
          </div>
        </Card>
      )}
      {m && m.calibrated === false && <Notice>{m.reason}</Notice>}
      <ExportPanel sessionId={id} synthetic={synthetic} />
      <Card title="Annotated video clip" subtitle="Overlays (boxes, skeleton, court lines, trail) burned into the video on the server. At most 60 s per clip.">
        <div className="flex flex-wrap items-end gap-4">
          <div className="w-36">
            <Field label="Start (s)">
              <input className={inputClass} type="number" min={0} step={0.5} value={start} onChange={(e) => setStart(e.target.value)} />
            </Field>
          </div>
          <div className="w-36">
            <Field label="End (s)">
              <input className={inputClass} type="number" min={0} step={0.5} value={end} onChange={(e) => setEnd(e.target.value)} />
            </Field>
          </div>
          <AnnotatedClipButton sessionId={id} range={() => range} />
        </div>
        <p className="mt-2 text-xs text-muted">
          Will render {fmtTime(range[0])} – {fmtTime(range[1])} ({fmtNum(range[1] - range[0], 1)} s). Rendering takes a few seconds.
        </p>
      </Card>
    </div>
  );
}
