import { useRef, useState } from "react";
import type { Feedback, ScoreCategory } from "../../shared/feedback";
import type { AnalysisResult } from "../lib/analyze";
import { TimelineChart } from "./TimelineChart";
import { VideoOverlay } from "./VideoOverlay";

const CATEGORY_LABEL: Record<ScoreCategory, string> = {
  footwork: "Footwork",
  ready_position: "Ready position",
  stroke_technique: "Stroke technique",
  power_generation: "Power generation",
  balance_recovery: "Balance & recovery",
};

const scoreTone = (s: number) => (s >= 8 ? "good" : s >= 5 ? "ok" : "low");

export function Report({ result, videoUrl }: { result: AnalysisResult; videoUrl: string }) {
  const { feedback, frames, samples, metrics, strokes, summary, hand, model } = result;
  const videoRef = useRef<HTMLVideoElement>(null);
  const [activeFrame, setActiveFrame] = useState<number | null>(null);

  const frameById = new Map(frames.map((f) => [f.id, f]));
  const noteById = new Map(feedback.frame_notes.map((n) => [n.frame_id, n.note]));

  const seek = (t: number) => {
    const video = videoRef.current;
    if (!video) return;
    video.pause();
    video.currentTime = t;
  };

  const showFrame = (id: number) => {
    const frame = frameById.get(id);
    if (!frame) return;
    setActiveFrame(id);
    seek(frame.timeSec);
    if (window.matchMedia("(max-width: 900px)").matches) {
      videoRef.current?.scrollIntoView({ behavior: "smooth", block: "center" });
    }
  };

  const frameChips = (ids: number[]) => {
    const valid = ids.filter((id) => frameById.has(id));
    if (!valid.length) return null;
    return (
      <div className="chips">
        {valid.map((id) => (
          <button key={id} className="chip" onClick={() => showFrame(id)} title={frameById.get(id)!.label}>
            Frame {id}
          </button>
        ))}
      </div>
    );
  };

  return (
    <div className="report">
      <aside className="report-media">
        <VideoOverlay videoUrl={videoUrl} videoRef={videoRef} samples={samples} hand={hand} />
        <TimelineChart
          samples={samples}
          metrics={metrics}
          strokes={strokes}
          frames={frames}
          duration={summary.durationSec}
          videoRef={videoRef}
          onSeek={seek}
        />
      </aside>

      <div className="report-body">
        <section className="card summary">
          <div className="summary-head">
            <span className="pill">{feedback.estimated_level}</span>
            <span className="muted">{feedback.detected_activity}</span>
          </div>
          <p className="summary-text">{feedback.summary}</p>
          <Scores scores={feedback.scores} />
        </section>

        {feedback.improvements.length > 0 && (
          <section className="card">
            <h2>What to work on</h2>
            <ol className="improvements">
              {feedback.improvements.map((item, i) => (
                <li key={i} className={`improvement ${item.priority}`}>
                  <div className="improvement-head">
                    <span className={`priority ${item.priority}`}>{item.priority}</span>
                    <h3>{item.title}</h3>
                  </div>
                  <p>{item.observation}</p>
                  <p className="why">
                    <strong>Why it matters:</strong> {item.why_it_matters}
                  </p>
                  <p className="fix">
                    <strong>How to fix it:</strong> {item.how_to_fix}
                  </p>
                  {frameChips(item.frame_ids)}
                </li>
              ))}
            </ol>
          </section>
        )}

        {feedback.strengths.length > 0 && (
          <section className="card">
            <h2>What you're doing well</h2>
            <ul className="strengths">
              {feedback.strengths.map((s, i) => (
                <li key={i}>
                  <h3>{s.title}</h3>
                  <p>{s.detail}</p>
                  {frameChips(s.frame_ids)}
                </li>
              ))}
            </ul>
          </section>
        )}

        {feedback.drills.length > 0 && (
          <section className="card">
            <h2>Drills</h2>
            <div className="drills">
              {feedback.drills.map((d, i) => (
                <article key={i} className="drill">
                  <h3>{d.name}</h3>
                  <p className="muted small">For: {d.targets}</p>
                  <p>{d.instructions}</p>
                  <p className="dosage">{d.dosage}</p>
                </article>
              ))}
            </div>
          </section>
        )}

        <section className="card">
          <h2>Key frames</h2>
          <div className="gallery">
            {frames.map((f) => (
              <button
                key={f.id}
                className={`frame${activeFrame === f.id ? " active" : ""}`}
                onClick={() => showFrame(f.id)}
              >
                <img src={f.dataUrl} alt={`Frame ${f.id}: ${f.label}`} loading="lazy" />
                <span className="frame-label">
                  <strong>#{f.id}</strong> {f.label} · {f.timeSec.toFixed(2)}s
                </span>
                {noteById.has(f.id) && <span className="frame-note">{noteById.get(f.id)}</span>}
              </button>
            ))}
          </div>
        </section>

        <details className="card measurements">
          <summary>Measurements</summary>
          <dl>
            <dt>Racket hand</dt>
            <dd>
              {summary.hand}
              {summary.handAutoDetected ? " (detected)" : ""}
            </dd>
            <dt>Swings detected</dt>
            <dd>{summary.strokes.length}</dd>
            <dt>Body tracked</dt>
            <dd>{Math.round(summary.poseCoverage * 100)}% of frames</dd>
            <dt>Median knee angle</dt>
            <dd>{summary.posture.medianKneeAngle ?? "n/a"}°</dd>
            <dt>Median stance width</dt>
            <dd>{summary.posture.medianStanceWidth ?? "n/a"}× shoulders</dd>
            <dt>Athletic stance</dt>
            <dd>
              {summary.posture.athleticStanceRatio !== null
                ? `${Math.round(summary.posture.athleticStanceRatio * 100)}% of the time`
                : "n/a"}
            </dd>
          </dl>
          <p className="muted small">
            Estimated from a single camera. Angles can be off by 10-15°, especially when a limb points toward the
            camera.
          </p>
        </details>

        {feedback.caveats.length > 0 && (
          <section className="caveats">
            <h2>Limits of this analysis</h2>
            <ul>
              {feedback.caveats.map((c, i) => (
                <li key={i}>{c}</li>
              ))}
            </ul>
          </section>
        )}

        <p className="muted small model-note">Feedback generated by {model}. Use it alongside a coach's eye.</p>
      </div>
    </div>
  );
}

function Scores({ scores }: { scores: Feedback["scores"] }) {
  if (!scores.length) return null;
  return (
    <ul className="scores">
      {scores.map((s) => {
        const score = Math.min(Math.max(s.score, 1), 10);
        return (
          <li key={s.category} title={s.rationale}>
            <div className="score-row">
              <span>{CATEGORY_LABEL[s.category]}</span>
              <span className={`score-num ${scoreTone(score)}`}>{score}/10</span>
            </div>
            <div className="score-bar">
              <div className={scoreTone(score)} style={{ width: `${score * 10}%` }} />
            </div>
            <p className="muted small">{s.rationale}</p>
          </li>
        );
      })}
    </ul>
  );
}
