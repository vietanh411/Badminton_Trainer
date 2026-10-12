import { useState, type DragEvent } from "react";
import type { Focus, HandPref, Level, PlayerContext } from "../../shared/types";
import { MAX_DURATION_SEC } from "../lib/analyze";

const MAX_BYTES = 500 * 1024 * 1024;

const FOCUS_OPTIONS: { value: Focus; label: string }[] = [
  { value: "general", label: "General technique" },
  { value: "overhead", label: "Overheads (clear / smash / drop)" },
  { value: "net", label: "Net play" },
  { value: "serve", label: "Serve" },
  { value: "footwork", label: "Footwork & movement" },
  { value: "defense", label: "Defence" },
];

interface Props {
  file: File | null;
  videoUrl: string | null;
  onFile: (file: File | null) => void;
  context: PlayerContext;
  onContextChange: (context: PlayerContext) => void;
  onStart: () => void;
}

export function UploadPanel({ file, videoUrl, onFile, context, onContextChange, onStart }: Props) {
  const [dragging, setDragging] = useState(false);
  const [fileError, setFileError] = useState<string | null>(null);
  const [duration, setDuration] = useState<number | null>(null);

  const accept = (f: File | undefined) => {
    setFileError(null);
    setDuration(null);
    if (!f) return;
    if (!f.type.startsWith("video/")) {
      setFileError("That doesn't look like a video file.");
      return;
    }
    if (f.size > MAX_BYTES) {
      setFileError("That file is over 500 MB. Trim it to a short clip first.");
      return;
    }
    onFile(f);
  };

  const onDrop = (e: DragEvent) => {
    e.preventDefault();
    setDragging(false);
    accept(e.dataTransfer.files[0]);
  };

  const tooLong = duration !== null && duration > MAX_DURATION_SEC;
  const set = <K extends keyof PlayerContext>(key: K, value: PlayerContext[K]) =>
    onContextChange({ ...context, [key]: value });

  if (!file || !videoUrl) {
    return (
      <section className="hero">
        <h1>Get coached on your badminton technique</h1>
        <p className="lede">
          Upload a short clip of yourself (or a player you coach). We track body movement frame by frame, then an AI
          coach reviews the key moments and tells you what to fix and how.
        </p>
        <label
          className={`dropzone${dragging ? " dragging" : ""}`}
          onDragOver={(e) => {
            e.preventDefault();
            setDragging(true);
          }}
          onDragLeave={() => setDragging(false)}
          onDrop={onDrop}
        >
          <input type="file" accept="video/*" onChange={(e) => accept(e.target.files?.[0])} hidden />
          <span className="dropzone-icon" aria-hidden>
            ⬆
          </span>
          <span className="dropzone-title">Drop a video here or click to choose</span>
          <span className="muted">MP4, MOV or WebM · 5-30 seconds works best · up to 2 minutes</span>
        </label>
        {fileError && <p className="field-error">{fileError}</p>}
        <ul className="tips">
          <li>
            <strong>Full body in frame</strong>, feet included, filmed from the side or behind the baseline.
          </li>
          <li>
            <strong>One player</strong> clearly visible. Other people in the shot can confuse the tracker.
          </li>
          <li>
            <strong>Steady camera</strong> at about chest height, in good light.
          </li>
        </ul>
      </section>
    );
  }

  return (
    <section className="setup">
      <div className="setup-video">
        <video
          src={videoUrl}
          controls
          playsInline
          muted
          onLoadedMetadata={(e) => setDuration(Number.isFinite(e.currentTarget.duration) ? e.currentTarget.duration : null)}
          onDurationChange={(e) => {
            if (Number.isFinite(e.currentTarget.duration)) setDuration(e.currentTarget.duration);
          }}
        />
        <div className="setup-file">
          <span className="truncate">{file.name}</span>
          {duration !== null && <span className="muted">{duration.toFixed(1)} s</span>}
          <button className="btn link" onClick={() => onFile(null)}>
            Change
          </button>
        </div>
        {tooLong && (
          <p className="field-error">
            This clip is {Math.round(duration!)} s long. Please trim it to under {MAX_DURATION_SEC / 60} minutes.
          </p>
        )}
      </div>

      <form
        className="setup-form"
        onSubmit={(e) => {
          e.preventDefault();
          onStart();
        }}
      >
        <h2>About the player</h2>

        <fieldset>
          <legend>Racket hand</legend>
          <div className="segmented">
            {(["auto", "right", "left"] as HandPref[]).map((h) => (
              <label key={h} className={context.hand === h ? "active" : ""}>
                <input type="radio" name="hand" checked={context.hand === h} onChange={() => set("hand", h)} />
                {h === "auto" ? "Detect" : h === "right" ? "Right" : "Left"}
              </label>
            ))}
          </div>
        </fieldset>

        <fieldset>
          <legend>Level</legend>
          <div className="segmented">
            {(["beginner", "intermediate", "advanced"] as Level[]).map((l) => (
              <label key={l} className={context.level === l ? "active" : ""}>
                <input type="radio" name="level" checked={context.level === l} onChange={() => set("level", l)} />
                {l[0].toUpperCase() + l.slice(1)}
              </label>
            ))}
          </div>
        </fieldset>

        <label className="field">
          <span>What do you want feedback on?</span>
          <select value={context.focus} onChange={(e) => set("focus", e.target.value as Focus)}>
            {FOCUS_OPTIONS.map((o) => (
              <option key={o.value} value={o.value}>
                {o.label}
              </option>
            ))}
          </select>
        </label>

        <label className="field">
          <span>
            Anything else? <span className="muted">(optional)</span>
          </span>
          <textarea
            rows={3}
            maxLength={600}
            placeholder="e.g. My clears don't reach the back line. This is shadow practice, no shuttle."
            value={context.notes}
            onChange={(e) => set("notes", e.target.value)}
          />
        </label>

        <button className="btn primary" type="submit" disabled={tooLong}>
          Analyze my technique
        </button>
        <p className="muted small">Takes about 1-2 minutes: tracking runs in your browser, then the coach reviews it.</p>
      </form>
    </section>
  );
}
