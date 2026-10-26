import { useMemo } from "react";
import { Area, AreaChart, Bar, BarChart, CartesianGrid, ReferenceArea, ReferenceLine, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { fmtNum, fmtTime, titleCase } from "../lib/format";
import type { Metrics, TrajectoryPoint } from "../types/api";

const AXIS = { stroke: "#5b6b88", fontSize: 11 };
const tooltipStyle = { background: "#070d18", border: "1px solid #22314d", borderRadius: 8, fontSize: 12, color: "#c9d4e5" };

export function SpeedChart({ points, time, interval, onSeek }: {
  points: TrajectoryPoint[];
  time: number;
  interval: [number, number] | null;
  onSeek?: (t: number) => void;
}) {
  const data = useMemo(() => points.map((p) => ({ t: p.t, speed: p.speed ?? null })), [points]);
  return (
    <ResponsiveContainer width="100%" height={220}>
      <AreaChart
        data={data}
        margin={{ top: 8, right: 8, bottom: 0, left: -18 }}
        onClick={(e) => {
          const label = (e as { activeLabel?: number | string } | null)?.activeLabel;
          if (onSeek && label !== undefined) onSeek(Number(label));
        }}
      >
        <defs>
          <linearGradient id="spd" x1="0" y1="0" x2="0" y2="1">
            <stop offset="0" stopColor="#22d3ee" stopOpacity={0.45} />
            <stop offset="1" stopColor="#22d3ee" stopOpacity={0} />
          </linearGradient>
        </defs>
        <CartesianGrid stroke="#1b2944" vertical={false} />
        <XAxis dataKey="t" type="number" domain={["dataMin", "dataMax"]} tickFormatter={(v: number) => fmtTime(v, 0)} {...AXIS} />
        <YAxis {...AXIS} unit="" tickFormatter={(v: number) => v.toFixed(1)} />
        <Tooltip
          contentStyle={tooltipStyle}
          labelFormatter={(v) => fmtTime(Number(v))}
          formatter={(v) => [`${fmtNum(Number(v), 2)} m/s`, "speed"]}
        />
        {interval && <ReferenceArea x1={interval[0]} x2={interval[1]} fill="#22d3ee" fillOpacity={0.08} stroke="#22d3ee" strokeOpacity={0.3} />}
        <ReferenceLine y={4} stroke="#fbbf24" strokeDasharray="4 4" strokeOpacity={0.5} />
        <Area type="monotone" dataKey="speed" stroke="#22d3ee" strokeWidth={1.5} fill="url(#spd)" connectNulls={false} isAnimationActive={false} dot={false} />
        <ReferenceLine x={time} stroke="#f8fafc" strokeOpacity={0.8} />
      </AreaChart>
    </ResponsiveContainer>
  );
}

export function SpeedHistogram({ hist }: { hist: NonNullable<Metrics["speed"]>["histogram"] }) {
  const data = hist.seconds.map((s, i) => ({ bin: `${hist.bin_edges_mps[i]}–${hist.bin_edges_mps[i + 1]}`, s }));
  return (
    <ResponsiveContainer width="100%" height={160}>
      <BarChart data={data} margin={{ top: 4, right: 4, bottom: 0, left: -24 }}>
        <CartesianGrid stroke="#1b2944" vertical={false} />
        <XAxis dataKey="bin" {...AXIS} interval={1} />
        <YAxis {...AXIS} />
        <Tooltip contentStyle={tooltipStyle} formatter={(v) => [`${fmtNum(Number(v), 1)} s`, "time"]} labelFormatter={(l) => `${l} m/s`} />
        <Bar dataKey="s" fill="#22d3ee" radius={[3, 3, 0, 0]} isAnimationActive={false} />
      </BarChart>
    </ResponsiveContainer>
  );
}

/** 3×3 player-relative zone grid: net at the top (front row), player's left on the left. */
export function ZoneGrid({ zones }: { zones: NonNullable<Metrics["zones"]> }) {
  const depths = ["front", "mid", "rear"];
  const lats = ["left", "center", "right"];
  const max = Math.max(0.0001, ...Object.values(zones.share));
  return (
    <div>
      <div className="mb-1 text-center text-[10px] uppercase tracking-[0.2em] text-muted">net</div>
      <div className="grid grid-cols-3 gap-1">
        {depths.flatMap((d) =>
          lats.map((l) => {
            const id = `${d}_${l}`;
            const share = zones.share[id] ?? 0;
            const visited = zones.visited.includes(id);
            return (
              <div
                key={id}
                title={`${titleCase(id)}: ${fmtNum(zones.time_s[id], 1)} s, ${zones.entries[id] ?? 0} entries`}
                className="relative flex aspect-[5/3] flex-col items-center justify-center rounded-md border text-center"
                style={{
                  background: `rgba(34,211,238,${0.06 + 0.6 * (share / max)})`,
                  borderColor: visited ? "rgba(163,230,53,0.45)" : "#22314d",
                }}
              >
                <div className="text-sm font-semibold tabular text-white">{fmtNum(share * 100, 0)}%</div>
                <div className="text-[10px] text-soft/80">{titleCase(id)}</div>
              </div>
            );
          }),
        )}
      </div>
      <div className="mt-1 text-center text-[10px] uppercase tracking-[0.2em] text-muted">baseline</div>
    </div>
  );
}

export function ShareBars({ shares, labels }: { shares: Record<string, number>; labels?: Record<string, string> }) {
  return (
    <div className="space-y-2">
      {Object.entries(shares).map(([k, v]) => (
        <div key={k}>
          <div className="mb-0.5 flex justify-between text-xs">
            <span className="text-muted">{labels?.[k] ?? titleCase(k)}</span>
            <span className="tabular text-soft">{fmtNum(v * 100, 0)}%</span>
          </div>
          <div className="h-1.5 rounded-full bg-navy-700">
            <div className="h-full rounded-full bg-gradient-to-r from-cyan-500 to-cyan-300" style={{ width: `${Math.min(100, v * 100)}%` }} />
          </div>
        </div>
      ))}
    </div>
  );
}
