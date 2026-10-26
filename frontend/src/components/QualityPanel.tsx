import { RefreshCw, UserCog } from "lucide-react";
import { useState } from "react";
import { useAsync } from "../hooks/useAsync";
import { DEF } from "../lib/definitions";
import { fmtNum, fmtTime, titleCase } from "../lib/format";
import { api } from "../services/api";
import type { Job, MetricsResponse, Session } from "../types/api";
import { JobProgress } from "./JobProgress";
import { Badge, Button, Card, ErrorBox, InfoTip, Notice } from "./ui";

function Row({ k, v, hint, tone }: { k: string; v: React.ReactNode; hint?: string; tone?: "good" | "warn" | "bad" }) {
  const cls = tone === "good" ? "text-good" : tone === "warn" ? "text-warn" : tone === "bad" ? "text-bad" : "text-soft";
  return (
    <div className="flex items-center justify-between border-b border-edge/50 py-1.5 text-sm last:border-0">
      <span className="flex items-center gap-1.5 text-muted">
        {k} {hint && <InfoTip text={hint} />}
      </span>
      <span className={`tabular ${cls}`}>{v}</span>
    </div>
  );
}

export function QualityPanel({ m }: { m: MetricsResponse }) {
  const q = m.quality;
  const cal = q.calibration;
  const dq = m.metrics.data_quality;
  return (
    <Card title="Data quality" subtitle="How much of this analysis is measured versus missing or inferred">
      <div className="grid gap-x-8 md:grid-cols-2">
        <div>
          <Row k="Target tracked" v={`${fmtNum(q.target_coverage_pct, 1)}% of frames`} hint={DEF.coverage} tone={q.target_coverage_pct > 85 ? "good" : q.target_coverage_pct > 60 ? "warn" : "bad"} />
          <Row k="Pose detected" v={`${fmtNum(m.metrics.pose?.pose_coverage_pct, 1)}% of tracked frames`} />
          <Row k="Occluded frames (box overlap)" v={q.occluded_frames} />
          <Row k="Identity links" v={q.identity_links} hint={DEF.identityLink} />
          <Row k="Possible ID switches" v={q.possible_id_switches} hint={DEF.idSwitch} tone={q.possible_id_switches ? "warn" : "good"} />
          <Row k="Tracking gaps ≥ 0.5 s" v={q.tracking_gaps} tone={q.tracking_gaps ? "warn" : "good"} />
          <Row k="People tracked" v={q.people_tracked} />
          <Row k="Player selection" v={q.selection.auto ? "automatic" : "chosen by user"} tone={q.selection.auto ? "warn" : undefined} />
          {q.shots && q.shots.cuts > 0 && (
            <>
              <Row
                k="Video cuts detected"
                v={`${q.shots.cuts} (${q.shots.count} shots)`}
                hint="Edited footage (e.g. highlights). The tracker restarts at every cut, nothing is connected across a cut, and the player is re-identified in each shot by appearance."
              />
              <Row
                k="Shots used / excluded"
                v={`${q.shots.used + q.shots.recalibrated} / ${q.shots.excluded}`}
                hint="Excluded shots do not show the calibrated court view (close-ups, replays, graphics, crowd)."
              />
              <Row k="Player re-identified after cuts" v={q.shots.cross_cut_links} />
              <Row
                k="Shots where the player was not found"
                v={q.shots.target_uncertain_shots}
                tone={q.shots.target_uncertain_shots ? "warn" : "good"}
                hint="Skipped rather than guessed when nobody, or more than one person, matched the player's appearance. Use an identity correction to assign the player in such a shot."
              />
            </>
          )}
        </div>
        <div>
          <Row k="Calibration source" v={cal.source ?? "none"} tone={cal.source ? (cal.source.includes("unverified") ? "warn" : "good") : "bad"} />
          <Row k="Calibration mean error" v={cal.mean_error_m !== undefined ? `${fmtNum(cal.mean_error_m * 100, 1)} cm` : "—"} hint={DEF.calibration} />
          {dq && (
            <>
              <Row k="Samples with position" v={`${dq.with_position} / ${dq.samples}`} />
              <Row k="Interpolated (short gaps)" v={dq.interpolated} />
              <Row k="Trajectory segments" v={dq.segments} />
              <Row k="Rejected positions" v={Object.entries(dq.rejected).map(([k, v]) => `${titleCase(k)} ${v}`).join(", ") || "none"} />
            </>
          )}
          <Row k="Detector / tracker" v={`${m.analysis.detector} / ${m.analysis.tracker}`} />
          <Row k="Mode · analysis fps" v={`${m.analysis.performance_mode} · ${fmtNum(m.analysis.fps, 0)}`} />
        </div>
      </div>
      {(cal.warnings?.length || cal.error) && (
        <div className="mt-3 space-y-1 text-xs text-warn">
          {cal.error && <div>• {cal.error}</div>}
          {cal.warnings?.map((w) => <div key={w}>• {w}</div>)}
        </div>
      )}
      <div className="mt-4 grid grid-cols-2 gap-2 text-xs sm:grid-cols-4">
        {Object.entries(m.timings_s).map(([k, v]) => (
          <div key={k} className="rounded-lg bg-navy-850 px-3 py-2">
            <div className="text-muted">{titleCase(k)}</div>
            <div className="tabular text-soft">{fmtNum(v, 2)} s</div>
          </div>
        ))}
        <div className="rounded-lg bg-navy-850 px-3 py-2">
          <div className="text-muted">Cache</div>
          <div className="text-soft">
            det {m.analysis.cache.detections ? "hit" : "miss"} · pose {m.analysis.cache.pose ? "hit" : "miss"}
          </div>
        </div>
      </div>
    </Card>
  );
}

export function IdentityPanel({ session, currentTime, onRerunDone }: { session: Session; currentTime: () => number; onRerunDone: () => void }) {
  const tracks = useAsync(() => api.tracks(session.id), [session.id]);
  const [corrections, setCorrections] = useState(session.corrections);
  const [job, setJob] = useState<string | null>(null);
  const [error, setError] = useState<unknown>(null);
  const t = currentTime();
  return (
    <Card
      title={
        <span className="flex items-center gap-2">
          <UserCog className="h-4 w-4 text-accent" /> Identity correction
        </span>
      }
      subtitle="If the tracker followed the wrong person, tell it who the target is from a given time, then re-run (detections are cached, so this is quick)."
    >
      {tracks.error ? <ErrorBox error={tracks.error} /> : null}
      <div className="max-h-64 overflow-y-auto">
        <table className="w-full text-xs">
          <thead className="text-left text-muted">
            <tr className="border-b border-edge">
              <th className="py-1.5 font-medium">Track</th>
              <th className="font-medium">Visible</th>
              <th className="font-medium">Frames</th>
              <th className="font-medium">Score</th>
              <th />
            </tr>
          </thead>
          <tbody>
            {(tracks.data ?? []).map((tr) => (
              <tr key={tr.track_id} className="border-b border-edge/50">
                <td className="py-1.5">
                  ID {tr.track_id} {tr.is_target && <Badge tone="accent">target</Badge>}
                </td>
                <td className="tabular text-soft">
                  {fmtTime(tr.first_t)} – {fmtTime(tr.last_t)}
                </td>
                <td className="tabular">{tr.n_frames}</td>
                <td className="tabular">{fmtNum(tr.mean_score, 2)}</td>
                <td className="text-right">
                  <button
                    className="text-accent hover:underline disabled:opacity-40"
                    disabled={t > tr.last_t + 0.5}
                    title={t > tr.last_t + 0.5 ? "This track ended before the current time" : undefined}
                    onClick={async () => {
                      try {
                        const r = await api.addCorrection(session.id, Math.max(tr.first_t, t), tr.track_id);
                        setCorrections(r.corrections);
                      } catch (e) {
                        setError(e);
                      }
                    }}
                  >
                    Target from {fmtTime(Math.max(tr.first_t, t))}
                  </button>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <div className="mt-3 flex flex-wrap items-center gap-2 text-xs">
        <Button
          variant="ghost"
          onClick={async () => {
            try {
              const r = await api.addCorrection(session.id, t, null);
              setCorrections(r.corrections);
            } catch (e) {
              setError(e);
            }
          }}
        >
          Target not visible from {fmtTime(t)}
        </Button>
        {corrections.length > 0 && (
          <Button
            variant="ghost"
            onClick={async () => {
              await api.clearCorrections(session.id);
              setCorrections([]);
            }}
          >
            Clear corrections
          </Button>
        )}
      </div>
      {corrections.length > 0 && (
        <div className="mt-2 text-xs text-muted">
          Pending corrections:{" "}
          {corrections.map((c) => `${fmtTime(c.t)} → ${c.track_id === null ? "not visible" : `ID ${c.track_id}`}`).join(" · ")}
        </div>
      )}
      {error ? <div className="mt-2"><ErrorBox error={error} /></div> : null}
      <div className="mt-4 border-t border-edge pt-4">
        {job ? (
          <JobProgress
            jobId={job}
            compact
            onDone={(j: Job) => {
              if (j.status === "COMPLETED") onRerunDone();
              setJob(null);
            }}
          />
        ) : (
          <Button
            variant="primary"
            onClick={async () => {
              try {
                setJob((await api.analyze(session.id, {})).id);
              } catch (e) {
                setError(e);
              }
            }}
          >
            <RefreshCw className="h-4 w-4" /> Re-run analysis with corrections
          </Button>
        )}
        <Notice tone="info">Detections and tracks are reused from the cache; only identity resolution, pose and analytics are recomputed.</Notice>
      </div>
    </Card>
  );
}
