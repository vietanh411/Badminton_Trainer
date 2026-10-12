import { useEffect, useState } from "react";
import type { Progress, Stage } from "../lib/analyze";

const STAGES: { stage: Stage; label: string }[] = [
  { stage: "loading", label: "Load pose model" },
  { stage: "tracking", label: "Track body movement" },
  { stage: "capturing", label: "Pick key moments" },
  { stage: "coaching", label: "Coach review" },
];

export function ProgressPanel({ progress, onCancel }: { progress: Progress; onCancel: () => void }) {
  const current = STAGES.findIndex((s) => s.stage === progress.stage);
  const [elapsed, setElapsed] = useState(0);

  useEffect(() => {
    setElapsed(0);
    const started = Date.now();
    const id = setInterval(() => setElapsed(Math.floor((Date.now() - started) / 1000)), 1000);
    return () => clearInterval(id);
  }, [progress.stage]);

  return (
    <section className="card progress" aria-live="polite">
      <h2>Analyzing your video</h2>
      <ol className="stages">
        {STAGES.map((s, i) => (
          <li key={s.stage} className={i < current ? "done" : i === current ? "active" : ""}>
            <span className="stage-dot" aria-hidden>
              {i < current ? "✓" : i + 1}
            </span>
            <span>{s.label}</span>
          </li>
        ))}
      </ol>

      <div className="progress-detail">
        <span>{progress.detail ?? STAGES[current]?.label}</span>
        <span className="muted">
          {progress.fraction !== null ? `${Math.round(progress.fraction * 100)}%` : `${elapsed}s`}
        </span>
      </div>
      <div
        className={`bar${progress.fraction === null ? " indeterminate" : ""}`}
        role="progressbar"
        aria-valuenow={progress.fraction !== null ? Math.round(progress.fraction * 100) : undefined}
      >
        <div style={{ width: progress.fraction !== null ? `${progress.fraction * 100}%` : undefined }} />
      </div>
      {progress.stage === "coaching" && (
        <p className="muted small">The coach looks closely at each frame. This usually takes 30-90 seconds.</p>
      )}

      <button className="btn ghost" onClick={onCancel}>
        Cancel
      </button>
    </section>
  );
}
