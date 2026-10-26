import { useRef, useState } from "react";
import { fmtTime } from "../lib/format";
import type { Stroke, TimelineEvent } from "../types/api";
import { cx } from "./ui";

export const EVENT_STYLE: Record<string, { color: string; label: string }> = {
  stroke: { color: "#22d3ee", label: "Stroke (candidate)" },
  direction_change: { color: "#a3e635", label: "Direction change" },
  high_speed: { color: "#fbbf24", label: "High-speed movement" },
  tracking_gap: { color: "#f87171", label: "Tracking gap" },
  possible_id_switch: { color: "#fb7185", label: "Possible ID switch" },
  identity_link: { color: "#a78bfa", label: "Identity link" },
  shot_cut: { color: "#e2e8f0", label: "Video cut" },
  shot_link: { color: "#c084fc", label: "Player re-identified after cut" },
  shot_target_uncertain: { color: "#fb923c", label: "Player not found after cut" },
  shot_excluded: { color: "#475569", label: "Not the court view (excluded)" },
};

export function Timeline({
  duration,
  time,
  events,
  strokes,
  interval,
  onSeek,
  onInterval,
}: {
  duration: number;
  time: number;
  events: TimelineEvent[];
  strokes: Stroke[];
  interval: [number, number] | null;
  onSeek: (t: number) => void;
  onInterval: (iv: [number, number] | null) => void;
}) {
  const ref = useRef<HTMLDivElement>(null);
  const [drag, setDrag] = useState<{ x0: number; t0: number; t1: number } | null>(null);
  const [hidden, setHidden] = useState<Set<string>>(new Set());
  const [hover, setHover] = useState<{ x: number; text: string } | null>(null);
  const d = Math.max(duration, 0.001);
  const tAt = (clientX: number) => {
    const r = ref.current!.getBoundingClientRect();
    return Math.min(d, Math.max(0, ((clientX - r.left) / r.width) * d));
  };
  const pct = (t: number) => `${(Math.min(Math.max(t, 0), d) / d) * 100}%`;

  const rangeTypes = new Set(["high_speed", "tracking_gap"]);
  const shotTypes = new Set(["shot_cut", "shot_excluded"]);
  const visible = events.filter(
    (e) => !hidden.has(e.type) && e.type !== "stroke" && !shotTypes.has(e.type) && EVENT_STYLE[e.type],
  );
  const excludedShots = hidden.has("shot_excluded") ? [] : events.filter((e) => e.type === "shot_excluded" && typeof e.t_end === "number");
  const cuts = hidden.has("shot_cut") ? [] : events.filter((e) => e.type === "shot_cut");
  const strokeVisible = !hidden.has("stroke");

  return (
    <div className="select-none">
      <div
        ref={ref}
        className="relative h-14 cursor-crosshair rounded-lg border border-edge bg-navy-850"
        onMouseDown={(e) => {
          const t = tAt(e.clientX);
          setDrag({ x0: e.clientX, t0: t, t1: t });
        }}
        onMouseMove={(e) => {
          if (drag) setDrag({ ...drag, t1: tAt(e.clientX) });
        }}
        onMouseUp={(e) => {
          if (!drag) return;
          if (Math.abs(e.clientX - drag.x0) < 4) onSeek(tAt(e.clientX));
          else onInterval([Math.min(drag.t0, drag.t1), Math.max(drag.t0, drag.t1)]);
          setDrag(null);
        }}
        onMouseLeave={() => {
          setDrag(null);
          setHover(null);
        }}
      >
        {/* tick marks */}
        {Array.from({ length: Math.floor(d / 5) + 1 }, (_, i) => i * 5).map((t) => (
          <div key={t} className="absolute bottom-0 h-1.5 w-px bg-edge" style={{ left: pct(t) }} />
        ))}
        {interval && (
          <div className="absolute inset-y-0 bg-cyan-400/15 ring-1 ring-cyan-400/40" style={{ left: pct(interval[0]), width: `calc(${pct(interval[1])} - ${pct(interval[0])})` }} />
        )}
        {drag && Math.abs(drag.t1 - drag.t0) > 0 && (
          <div className="absolute inset-y-0 bg-cyan-400/20" style={{ left: pct(Math.min(drag.t0, drag.t1)), width: `${(Math.abs(drag.t1 - drag.t0) / d) * 100}%` }} />
        )}
        {/* Excluded shots (no calibrated court view) are greyed out; cuts are thin full-height lines. */}
        {excludedShots.map((e, i) => (
          <div
            key={`x${i}`}
            className="absolute inset-y-0 bg-[repeating-linear-gradient(135deg,rgba(71,85,105,0.55)_0_4px,transparent_4px_8px)]"
            style={{ left: pct(e.t), width: `max(2px, calc(${pct(e.t_end as number)} - ${pct(e.t)}))` }}
            onMouseEnter={(ev) => setHover({ x: ev.clientX, text: `Excluded ${fmtTime(e.t)}–${fmtTime(e.t_end)}: ${String(e.reason ?? "")}` })}
          />
        ))}
        {cuts.map((e, i) => (
          <div
            key={`c${i}`}
            className="absolute inset-y-0 w-px bg-slate-200/50"
            style={{ left: pct(e.t) }}
            onMouseEnter={(ev) => setHover({ x: ev.clientX, text: `Video cut @ ${fmtTime(e.t)}` })}
          />
        ))}
        {visible.map((e, i) => {
          const st = EVENT_STYLE[e.type];
          const isRange = rangeTypes.has(e.type) && typeof e.t_end === "number";
          const row = e.type === "direction_change" ? "top-[60%]" : e.type === "high_speed" ? "top-[42%]" : "top-[78%]";
          return isRange ? (
            <div
              key={i}
              className={cx("absolute h-1.5 rounded-full", row)}
              style={{ left: pct(e.t), width: `max(3px, calc(${pct(e.t_end as number)} - ${pct(e.t)}))`, background: st.color, opacity: 0.8 }}
              onMouseEnter={(ev) => setHover({ x: ev.clientX, text: `${st.label} ${fmtTime(e.t)}–${fmtTime(e.t_end)}` })}
            />
          ) : (
            <div
              key={i}
              className={cx("absolute h-2 w-2 -translate-x-1/2 rounded-full", row)}
              style={{ left: pct(e.t), background: st.color }}
              onMouseEnter={(ev) => setHover({ x: ev.clientX, text: `${st.label} @ ${fmtTime(e.t)}` })}
            />
          );
        })}
        {strokeVisible &&
          strokes.map((s) => (
            <div
              key={s.id}
              className={cx("absolute top-1 h-5 w-0.5 -translate-x-1/2 rounded", s.review_status === "rejected" && "opacity-30")}
              style={{ left: pct(s.t), background: s.source === "manual" ? "#e879f9" : "#22d3ee" }}
              onMouseEnter={(ev) => setHover({ x: ev.clientX, text: `Stroke ${s.source} (${s.review_status}) @ ${fmtTime(s.t)}` })}
            />
          ))}
        <div className="pointer-events-none absolute inset-y-0 w-0.5 bg-white shadow-[0_0_8px_rgba(255,255,255,0.7)]" style={{ left: pct(time) }} />
        {hover && (
          <div
            className="pointer-events-none fixed z-40 -translate-x-1/2 -translate-y-full rounded border border-edge bg-ink px-2 py-1 text-[11px] text-soft"
            style={{ left: hover.x, top: (ref.current?.getBoundingClientRect().top ?? 0) - 4 }}
          >
            {hover.text}
          </div>
        )}
      </div>
      <div className="mt-2 flex flex-wrap items-center gap-x-4 gap-y-1 text-[11px] text-muted">
        <span className="tabular text-soft">
          {fmtTime(time)} / {fmtTime(duration)}
        </span>
        {Object.entries(EVENT_STYLE).map(([type, st]) => (
          <button
            key={type}
            onClick={() => {
              const n = new Set(hidden);
              if (n.has(type)) n.delete(type);
              else n.add(type);
              setHidden(n);
            }}
            className={cx("flex items-center gap-1.5 transition", hidden.has(type) && "opacity-40")}
          >
            <span className="h-2 w-2 rounded-full" style={{ background: st.color }} />
            {st.label}
          </button>
        ))}
        {interval ? (
          <button className="ml-auto text-accent hover:underline" onClick={() => onInterval(null)}>
            Clear selection ({fmtTime(interval[0])}–{fmtTime(interval[1])})
          </button>
        ) : (
          <span className="ml-auto">Click to seek · drag to select an interval</span>
        )}
      </div>
    </div>
  );
}
