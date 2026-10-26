import { Check, Plus, Trash2, X } from "lucide-react";
import { useState } from "react";
import { DEF } from "../lib/definitions";
import { fmtNum, fmtTime, titleCase } from "../lib/format";
import { api } from "../services/api";
import { SHOT_LABELS, type Metrics, type Stroke } from "../types/api";
import { Badge, Button, Card, cx, EmptyState, ErrorBox, ExperimentalBadge, InfoTip, Notice } from "./ui";

const selectClass = "rounded-md border border-edge bg-navy-850 px-2 py-1 text-xs text-soft outline-none focus:border-accent";

export function StrokePanel({
  sessionId,
  strokes,
  setStrokes,
  info,
  currentTime,
  onSeek,
}: {
  sessionId: string;
  strokes: Stroke[];
  setStrokes: (update: (prev: Stroke[]) => Stroke[]) => void;
  info: Metrics["strokes"];
  currentTime: () => number;
  onSeek: (t: number) => void;
}) {
  const [error, setError] = useState<unknown>(null);
  const [filter, setFilter] = useState<"all" | "candidate" | "confirmed" | "rejected">("all");

  async function patch(s: Stroke, body: Parameters<typeof api.patchStroke>[1]) {
    // Optimistic, field-scoped update: concurrent PATCHes may return in any order, so only the
    // fields this request changed are taken from its response.
    const keys = Object.keys(body) as (keyof Stroke)[];
    setStrokes((prev) => prev.map((x) => (x.id === s.id ? { ...x, ...body } as Stroke : x)));
    try {
      const n = await api.patchStroke(s.id, body);
      setStrokes((prev) =>
        prev.map((x) => (x.id === s.id ? ({ ...x, ...Object.fromEntries(keys.map((k) => [k, n[k]])) } as Stroke) : x)),
      );
    } catch (e) {
      setStrokes((prev) => prev.map((x) => (x.id === s.id ? s : x)));
      setError(e);
    }
  }

  const shown = strokes.filter((s) => filter === "all" || s.review_status === filter);
  const counts = {
    candidate: strokes.filter((s) => s.review_status === "candidate").length,
    confirmed: strokes.filter((s) => s.review_status === "confirmed").length,
    rejected: strokes.filter((s) => s.review_status === "rejected").length,
  };

  return (
    <Card
      title={
        <span className="flex items-center gap-2">
          Stroke events <ExperimentalBadge /> <InfoTip text={DEF.strokes} />
        </span>
      }
      subtitle="Review detected candidates and label them to build a training set for the optional classifier."
      actions={
        <Button
          onClick={async () => {
            try {
              const s = await api.addStroke(sessionId, { t: Math.round(currentTime() * 1000) / 1000 });
              setStrokes((prev) => [...prev, s].sort((a, b) => a.t - b.t));
            } catch (e) {
              setError(e);
            }
          }}
        >
          <Plus className="h-4 w-4" /> Add stroke at {fmtTime(currentTime())}
        </Button>
      }
    >
      <div className="mb-4 space-y-2">
        <Notice tone="info">
          Detection uses racket-wrist speed peaks. <b>Height</b> (overhead / side / underarm) and <b>side</b> (forehand / backhand) are
          transparent pose heuristics. Smash vs clear vs drop is <b>not</b> predicted: without shuttle tracking the arm action alone is
          not reliable evidence. Labels you add here are human annotations.
        </Notice>
        {info?.warning && <Notice>{info.warning}</Notice>}
        <div className="flex flex-wrap items-center gap-2 text-xs text-muted">
          <span>Handedness: <b className="text-soft">{info?.handedness ?? "—"}</b></span>
          <span>· sampling {fmtNum(info?.sampling_fps, 0)} fps</span>
          {info?.model && Object.keys(info.model).length ? <span>· classifier: {Object.values(info.model).join(", ")}</span> : <span>· classifier: heuristic-v1</span>}
          <span className="ml-auto flex gap-1">
            {(["all", "candidate", "confirmed", "rejected"] as const).map((f) => (
              <button key={f} onClick={() => setFilter(f)} className={cx("rounded px-2 py-0.5", filter === f ? "bg-navy-600 text-accent" : "hover:text-soft")}>
                {titleCase(f)} {f !== "all" && `(${counts[f]})`}
              </button>
            ))}
          </span>
        </div>
      </div>
      {error ? <div className="mb-3"><ErrorBox error={error} /></div> : null}
      {shown.length === 0 ? (
        <EmptyState title="No stroke events">Play the video and add strokes manually at the moment of contact, or re-run with stroke detection enabled.</EmptyState>
      ) : (
        <div className="overflow-x-auto">
          <table className="w-full min-w-[820px] text-xs">
            <thead className="text-left text-muted">
              <tr className="border-b border-edge">
                <th className="py-2 font-medium">Time</th>
                <th className="font-medium">Source</th>
                <th className="font-medium">Detection conf.</th>
                <th className="font-medium">Height (heuristic)</th>
                <th className="font-medium">Side (heuristic)</th>
                <th className="font-medium">Shot (heuristic)</th>
                <th className="font-medium">Your label</th>
                <th className="text-right font-medium">Review</th>
              </tr>
            </thead>
            <tbody>
              {shown.map((s) => (
                <tr key={s.id} className={cx("border-b border-edge/50 transition hover:bg-navy-700/40", s.review_status === "rejected" && "opacity-50")}>
                  <td className="py-2">
                    <button className="font-mono text-accent hover:underline" onClick={() => onSeek(Math.max(0, (s.t_start ?? s.t - 0.5)))}>
                      {fmtTime(s.t, 2)}
                    </button>
                  </td>
                  <td>
                    <Badge tone={s.source === "manual" ? "accent" : "neutral"}>{s.source}</Badge>
                  </td>
                  <td className="tabular">{s.confidence !== null ? fmtNum(s.confidence * 100, 0) + "%" : "—"}</td>
                  <td className="text-soft">{s.height_class ?? "—"}</td>
                  <td className="text-soft">
                    {s.stroke_side ?? "—"}
                    {s.stroke_side_confidence ? <span className="text-muted"> ({fmtNum(s.stroke_side_confidence * 100, 0)}%)</span> : null}
                  </td>
                  <td className="text-soft" title={s.classifier ?? undefined}>
                    {s.shot_type ? titleCase(s.shot_type.replace("_unspecified", " (type unknown)")) : "—"}
                  </td>
                  <td>
                    <div className="flex gap-1">
                      <select
                        className={selectClass}
                        value={s.label_shot_type ?? ""}
                        onChange={(e) => patch(s, { label_shot_type: e.target.value || null })}
                        aria-label="Shot type label"
                      >
                        <option value="">shot…</option>
                        {SHOT_LABELS.map((l) => (
                          <option key={l} value={l}>
                            {titleCase(l)}
                          </option>
                        ))}
                      </select>
                      <select
                        className={selectClass}
                        value={s.label_stroke_side ?? ""}
                        onChange={(e) => patch(s, { label_stroke_side: e.target.value || null })}
                        aria-label="Stroke side label"
                      >
                        <option value="">side…</option>
                        <option value="forehand">Forehand</option>
                        <option value="backhand">Backhand</option>
                      </select>
                    </div>
                  </td>
                  <td>
                    <div className="flex justify-end gap-1">
                      <button
                        title="Confirm"
                        onClick={() => patch(s, { review_status: s.review_status === "confirmed" ? "candidate" : "confirmed" })}
                        className={cx("rounded p-1", s.review_status === "confirmed" ? "bg-lime-400/20 text-good" : "text-muted hover:text-good")}
                      >
                        <Check className="h-4 w-4" />
                      </button>
                      <button
                        title="Reject (false detection)"
                        onClick={() => patch(s, { review_status: s.review_status === "rejected" ? "candidate" : "rejected" })}
                        className={cx("rounded p-1", s.review_status === "rejected" ? "bg-red-400/20 text-bad" : "text-muted hover:text-bad")}
                      >
                        <X className="h-4 w-4" />
                      </button>
                      <button
                        title="Delete"
                        onClick={async () => {
                          try {
                            await api.deleteStroke(s.id);
                            setStrokes((prev) => prev.filter((x) => x.id !== s.id));
                          } catch (e) {
                            setError(e);
                          }
                        }}
                        className="rounded p-1 text-muted hover:text-bad"
                      >
                        <Trash2 className="h-4 w-4" />
                      </button>
                    </div>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </Card>
  );
}
