import { Check, CircleX, Loader2 } from "lucide-react";
import { useState } from "react";
import { useJob } from "../hooks/useJob";
import { fmtNum, titleCase } from "../lib/format";
import { api } from "../services/api";
import { JOB_STAGES, type Job } from "../types/api";
import { Button, cx, ErrorBox, Notice } from "./ui";

const SHORT: Record<string, string> = {
  QUEUED: "Queued",
  VALIDATING: "Validate",
  CALIBRATING: "Calibrate",
  DETECTING: "Detect",
  TRACKING: "Track",
  ESTIMATING_POSE: "Pose",
  ANALYZING: "Analyse",
  RENDERING: "Render",
  COMPLETED: "Done",
};

const HINTS: [RegExp, string][] = [
  [/match|selected/i, "Re-open the setup, pick a frame where the player is clearly visible and click their box again."],
  [/model|download/i, "Run `python scripts/download_models.py` on the server to fetch the model weights."],
  [/calibration|court/i, "Check the court calibration (landmarks on the correct lines) and re-run."],
  [/restart/i, "Start the analysis again."],
];

export function JobProgress({ jobId, onDone, compact }: { jobId: string; onDone?: (job: Job) => void; compact?: boolean }) {
  const job = useJob(jobId, onDone);
  const [cancelling, setCancelling] = useState(false);
  if (!job)
    return (
      <div className="flex items-center gap-2 text-sm text-muted">
        <Loader2 className="h-4 w-4 animate-spin text-accent" /> Connecting to job…
      </div>
    );
  const idx = JOB_STAGES.indexOf(job.status);
  const failed = job.status === "FAILED" || job.status === "CANCELLED";
  const hint = job.error ? HINTS.find(([re]) => re.test(job.error ?? ""))?.[1] : undefined;
  return (
    <div className="space-y-4">
      {!compact && (
        <ol className="grid grid-cols-3 gap-2 sm:grid-cols-5 lg:grid-cols-9">
          {JOB_STAGES.map((st, i) => {
            const done = idx > i || job.status === "COMPLETED";
            const active = idx === i && !failed && job.status !== "COMPLETED";
            return (
              <li key={st} className="flex flex-col items-center gap-1.5 text-center">
                <span
                  className={cx(
                    "grid h-7 w-7 place-items-center rounded-full border text-[11px] font-semibold transition",
                    done && "border-lime-400/50 bg-lime-400/15 text-good",
                    active && "border-accent bg-cyan-400/15 text-accent shadow-[0_0_14px_rgba(34,211,238,0.5)]",
                    !done && !active && "border-edge text-muted",
                  )}
                >
                  {done ? <Check className="h-3.5 w-3.5" /> : active ? <Loader2 className="h-3.5 w-3.5 animate-spin" /> : i + 1}
                </span>
                <span className={cx("text-[10px] font-medium uppercase tracking-wide", active ? "text-accent" : "text-muted")} title={titleCase(st.toLowerCase())}>
                  {SHORT[st]}
                </span>
              </li>
            );
          })}
        </ol>
      )}
      <div>
        <div className="mb-1.5 flex items-center justify-between text-xs">
          <span className={cx(failed ? "text-bad" : job.status === "COMPLETED" ? "text-good" : "text-soft")}>
            {job.status === "COMPLETED" ? "Analysis complete" : failed ? titleCase(job.status.toLowerCase()) : job.message ?? titleCase(job.status.toLowerCase())}
          </span>
          <span className="tabular text-muted">{fmtNum(job.progress * 100, 0)}%</span>
        </div>
        <div className="h-2 overflow-hidden rounded-full bg-navy-700">
          <div
            className={cx("h-full rounded-full transition-all duration-500", failed ? "bg-bad" : job.status === "COMPLETED" ? "bg-good" : "bg-accent")}
            style={{ width: `${Math.max(2, job.progress * 100)}%` }}
          />
        </div>
      </div>
      {job.status === "FAILED" && (
        <ErrorBox error={job.error ?? "The analysis failed."} action={hint ? <span className="text-xs text-red-100/80">{hint}</span> : undefined} />
      )}
      {job.status === "CANCELLED" && (
        <Notice tone="info">
          <CircleX className="mr-1 inline h-4 w-4" /> The analysis was cancelled. You can change settings and start it again.
        </Notice>
      )}
      {!failed && job.status !== "COMPLETED" && (
        <Button
          variant="danger"
          loading={cancelling}
          onClick={async () => {
            setCancelling(true);
            try {
              await api.cancelJob(job.id);
            } catch {
              /* already finished */
            } finally {
              setCancelling(false);
            }
          }}
        >
          Cancel analysis
        </Button>
      )}
    </div>
  );
}
