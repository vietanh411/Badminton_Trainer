import { ArrowRight, Check, Play } from "lucide-react";
import { useEffect, useState } from "react";
import { Link, useNavigate, useParams, useSearchParams } from "react-router-dom";
import { CalibrationEditor } from "../components/CalibrationEditor";
import { JobProgress } from "../components/JobProgress";
import { PlayerSelector } from "../components/PlayerSelector";
import { Button, Card, cx, ErrorBox, Field, inputClass, Notice, Segmented, Spinner, SyntheticBanner, Toggle } from "../components/ui";
import { useAsync } from "../hooks/useAsync";
import { useSystem } from "../hooks/useStatic";
import { fmtBytes, fmtDuration, fmtNum } from "../lib/format";
import { api, mediaUrl } from "../services/api";
import type { Session } from "../types/api";

type Step = "video" | "calibration" | "player" | "run";
type Mode = "fast" | "balanced" | "accurate";

function StepTab({ active, done, n, label, onClick }: { active: boolean; done: boolean; n: number; label: string; onClick: () => void }) {
  return (
    <button
      onClick={onClick}
      className={cx(
        "flex flex-1 items-center gap-3 rounded-xl border px-4 py-3 text-left transition",
        active ? "border-cyan-400/50 bg-cyan-400/10" : "border-edge bg-navy-850 hover:border-cyan-400/30",
      )}
    >
      <span
        className={cx(
          "grid h-7 w-7 shrink-0 place-items-center rounded-full text-xs font-bold",
          done ? "bg-lime-400/20 text-good" : active ? "bg-accent text-navy-900" : "bg-navy-700 text-muted",
        )}
      >
        {done ? <Check className="h-4 w-4" /> : n}
      </span>
      <span className={cx("text-sm font-medium", active ? "text-white" : "text-soft")}>{label}</span>
    </button>
  );
}

export function SessionSetup() {
  const { id = "" } = useParams();
  const [search] = useSearchParams();
  const navigate = useNavigate();
  const sys = useSystem();
  const sess = useAsync(() => api.session(id), [id]);
  const s = sess.data;
  const [step, setStep] = useState<Step>("video");
  const [jobId, setJobId] = useState<string | null>(search.get("job"));
  const [mode, setMode] = useState<Mode>("balanced");
  const [detector, setDetector] = useState("auto");
  const [tracker, setTracker] = useState("bytetrack");
  const [fps, setFps] = useState("");
  const [strokes, setStrokes] = useState(true);
  const [starting, setStarting] = useState(false);
  const [error, setError] = useState<unknown>(null);

  useEffect(() => {
    if (!s) return;
    if (jobId) setStep("run");
    else if (s.latest_job && ["QUEUED", "VALIDATING", "CALIBRATING", "DETECTING", "TRACKING", "ESTIMATING_POSE", "ANALYZING", "RENDERING"].includes(s.latest_job.status)) {
      setJobId(s.latest_job.id);
      setStep("run");
    }
    const st = s.settings;
    if (st.performance_mode) setMode(st.performance_mode);
    if (st.detector && st.detector !== "truth") setDetector(st.detector);
    if (st.tracker) setTracker(st.tracker);
    if (typeof st.detect_strokes === "boolean") setStrokes(st.detect_strokes);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [s?.id]);

  if (sess.error) return <ErrorBox error={sess.error} />;
  if (!s) return <Spinner label="Loading session…" />;
  const synthetic = s.data_source === "synthetic" || s.video.is_synthetic;

  async function analyze() {
    setStarting(true);
    setError(null);
    try {
      const body: Record<string, unknown> = { performance_mode: mode, tracker, detect_strokes: strokes };
      if (!synthetic) body.detector = detector;
      if (fps) body.analysis_fps = Number(fps);
      const job = await api.analyze(id, body);
      setJobId(job.id);
    } catch (e) {
      setError(e);
    } finally {
      setStarting(false);
    }
  }

  const update = (n: Session) => sess.setData(n);

  return (
    <div className="space-y-6">
      <div className="flex flex-wrap items-end justify-between gap-3">
        <div>
          <div className="text-xs font-medium uppercase tracking-wider text-accent">Session setup</div>
          <h1 className="mt-1 text-2xl font-bold tracking-tight text-white">{s.name}</h1>
        </div>
        {s.status === "completed" && (
          <Link to={`/sessions/${id}`} className="inline-flex items-center gap-1 text-sm text-accent hover:underline">
            View analysis <ArrowRight className="h-4 w-4" />
          </Link>
        )}
      </div>
      {synthetic && <SyntheticBanner />}
      <div className="flex flex-col gap-2 md:flex-row">
        <StepTab n={1} label="Video" active={step === "video"} done onClick={() => setStep("video")} />
        <StepTab n={2} label="Court calibration" active={step === "calibration"} done={!!s.calibration} onClick={() => setStep("calibration")} />
        <StepTab n={3} label="Player selection" active={step === "player"} done={!!s.player_selection} onClick={() => setStep("player")} />
        <StepTab n={4} label="Options & analysis" active={step === "run"} done={s.status === "completed"} onClick={() => setStep("run")} />
      </div>

      {step === "video" && (
        <div className="grid gap-5 lg:grid-cols-[minmax(0,1fr)_320px]">
          <Card title="Preview">
            <video src={mediaUrl(`/api/videos/${s.video.id}/preview`)} controls muted className="aspect-video w-full rounded-xl bg-black" />
            {!s.video.browser_playable && !s.video.has_proxy && (
              <div className="mt-3">
                <Notice>
                  Codec <b>{s.video.codec}</b> may not play in your browser. Calibration and player selection use server-decoded frames, and a
                  playable preview is generated during analysis.
                </Notice>
              </div>
            )}
          </Card>
          <Card title="Metadata">
            <dl className="grid grid-cols-2 gap-y-2 text-sm">
              {[
                ["File", s.video.original_filename],
                ["Resolution", `${s.video.width}×${s.video.height}`],
                ["Frame rate", `${fmtNum(s.video.fps, 2)} fps`],
                ["Duration", fmtDuration(s.video.duration_s)],
                ["Frames", s.video.frame_count.toLocaleString()],
                ["Codec", `${s.video.codec} (${s.video.container})`],
                ["Rotation", `${s.video.rotation}°`],
                ["Size", fmtBytes(s.video.size_bytes)],
              ].map(([k, v]) => (
                <div key={k} className="contents">
                  <dt className="text-muted">{k}</dt>
                  <dd className="truncate text-right text-soft" title={String(v)}>
                    {v}
                  </dd>
                </div>
              ))}
            </dl>
            <Button variant="primary" className="mt-5 w-full" onClick={() => setStep("calibration")}>
              Calibrate the court <ArrowRight className="h-4 w-4" />
            </Button>
          </Card>
        </div>
      )}

      {step === "calibration" && (
        <>
          <CalibrationEditor session={s} onSaved={(c) => update({ ...s, calibration: c })} />
          <div className="flex justify-end">
            <Button variant="primary" onClick={() => setStep("player")}>
              Next: choose the player <ArrowRight className="h-4 w-4" />
            </Button>
          </div>
        </>
      )}

      {step === "player" && (
        <>
          <PlayerSelector session={s} onSelected={update} />
          <div className="flex justify-end">
            <Button variant="primary" onClick={() => setStep("run")}>
              Next: options <ArrowRight className="h-4 w-4" />
            </Button>
          </div>
        </>
      )}

      {step === "run" && (
        <div className="grid gap-5 lg:grid-cols-[minmax(0,1fr)_minmax(0,1fr)]">
          <Card title="Processing options" subtitle="Expensive stages are cached: re-running with different analytics settings is fast.">
            <div className="space-y-5">
              <Field label="Performance mode" hint="Trade-off between speed and accuracy. Higher modes sample more frames at higher resolution with larger models.">
                <Segmented<Mode>
                  value={mode}
                  onChange={setMode}
                  options={[
                    { value: "fast", label: "Fast" },
                    { value: "balanced", label: "Balanced" },
                    { value: "accurate", label: "High accuracy" },
                  ]}
                />
                <div className="mt-1.5 text-xs text-muted">{sys?.profiles[mode]?.description}</div>
              </Field>
              {synthetic ? (
                <Notice tone="info">Synthetic sessions are analysed with the simulator's ground-truth detections and pose.</Notice>
              ) : (
                <div className="grid gap-4 sm:grid-cols-2">
                  <Field label="Person detector" hint="Auto uses YOLO (Ultralytics, AGPL-3.0) when installed, otherwise MediaPipe EfficientDet (Apache-2.0).">
                    <select className={inputClass} value={detector} onChange={(e) => setDetector(e.target.value)}>
                      <option value="auto">Auto</option>
                      <option value="yolo">YOLO11</option>
                      <option value="mediapipe">MediaPipe EfficientDet</option>
                    </select>
                  </Field>
                  <Field label="Tracker" hint="ByteTrack is built in. BoT-SORT uses Ultralytics' implementation and requires the YOLO detector.">
                    <select className={inputClass} value={tracker} onChange={(e) => setTracker(e.target.value)}>
                      <option value="bytetrack">ByteTrack</option>
                      <option value="botsort">BoT-SORT</option>
                    </select>
                  </Field>
                </div>
              )}
              <div className="grid gap-4 sm:grid-cols-2">
                <Field label="Analysis FPS override" hint="Leave empty to use the mode's default. ≥ 25 fps resolves fast racket swings better.">
                  <input className={inputClass} type="number" min={1} max={60} placeholder={String(sys?.profiles[mode]?.analysis_fps ?? "")} value={fps} onChange={(e) => setFps(e.target.value)} />
                </Field>
                <div className="flex items-end pb-2">
                  <Toggle checked={strokes} onChange={setStrokes} label="Detect stroke events (experimental)" />
                </div>
              </div>
              <div className="space-y-1 rounded-lg border border-edge bg-navy-850 p-3 text-xs">
                <div className={s.calibration ? "text-good" : "text-warn"}>
                  {s.calibration
                    ? `✓ Court calibrated (${s.calibration.method}, mean error ${fmtNum(s.calibration.mean_error_m * 100, 1)} cm)`
                    : "! No calibration: automatic court detection will be tried; if it fails, metric court analytics are disabled."}
                </div>
                <div className={s.player_selection ? "text-good" : "text-warn"}>
                  {s.player_selection ? "✓ Player selected" : "! No player selected: the longest-tracked near-court person is used."}
                </div>
              </div>
              {error ? <ErrorBox error={error} /> : null}
              <Button variant="primary" className="w-full" onClick={analyze} loading={starting} disabled={!!jobId && s.status === "processing"}>
                <Play className="h-4 w-4" /> {s.status === "completed" ? "Re-run analysis" : "Start analysis"}
              </Button>
            </div>
          </Card>
          <Card title="Progress">
            {jobId ? (
              <JobProgress
                key={jobId}
                jobId={jobId}
                onDone={(j) => {
                  sess.reload();
                  if (j.status === "COMPLETED") window.setTimeout(() => navigate(`/sessions/${id}`), 900);
                }}
              />
            ) : (
              <div className="text-sm text-muted">Start the analysis to see live progress through each pipeline stage.</div>
            )}
          </Card>
        </div>
      )}
    </div>
  );
}
