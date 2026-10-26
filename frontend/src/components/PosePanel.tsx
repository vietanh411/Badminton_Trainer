import { useMemo, useState } from "react";
import { CartesianGrid, Legend, Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { useAsync } from "../hooks/useAsync";
import { fmtNum, fmtTime, titleCase } from "../lib/format";
import { poseApi, type PoseSeries } from "../services/api";
import { Card, ErrorBox, Notice, Segmented, Skeleton } from "./ui";

const LANDMARKS = [
  "right_wrist", "left_wrist", "right_elbow", "left_elbow", "right_shoulder", "left_shoulder",
  "right_hip", "left_hip", "right_knee", "left_knee", "right_ankle", "left_ankle", "nose",
];
const ANGLES: { key: string; label: string; unit: string }[] = [
  { key: "right_knee_flexion", label: "Right knee flexion", unit: "°" },
  { key: "left_knee_flexion", label: "Left knee flexion", unit: "°" },
  { key: "right_elbow_angle", label: "Right elbow angle", unit: "°" },
  { key: "left_elbow_angle", label: "Left elbow angle", unit: "°" },
  { key: "torso_lean_deg", label: "Torso lean", unit: "°" },
  { key: "stance_width", label: "Stance width", unit: "torso lengths" },
];
const AXIS = { stroke: "#5b6b88", fontSize: 11 };
const tooltipStyle = { background: "#070d18", border: "1px solid #22314d", borderRadius: 8, fontSize: 12, color: "#c9d4e5" };

function series(d: PoseSeries, key: string): (number | null)[] {
  const v = d[key];
  return Array.isArray(v) ? (v as (number | null)[]) : [];
}

function RawSmoothChart({ t, raw, smooth, unit, onSeek }: { t: number[]; raw: (number | null)[]; smooth: (number | null)[]; unit: string; onSeek: (t: number) => void }) {
  const data = useMemo(() => {
    const step = Math.max(1, Math.ceil(t.length / 2500));
    const out: { t: number; raw: number | null; smooth: number | null }[] = [];
    for (let i = 0; i < t.length; i += step) out.push({ t: t[i], raw: raw[i] ?? null, smooth: smooth[i] ?? null });
    return out;
  }, [t, raw, smooth]);
  return (
    <ResponsiveContainer width="100%" height={220}>
      <LineChart
        data={data}
        margin={{ top: 8, right: 8, bottom: 0, left: -10 }}
        onClick={(e) => {
          const label = (e as { activeLabel?: number | string } | null)?.activeLabel;
          if (label !== undefined) onSeek(Number(label));
        }}
      >
        <CartesianGrid stroke="#1b2944" vertical={false} />
        <XAxis dataKey="t" type="number" domain={["dataMin", "dataMax"]} tickFormatter={(v: number) => fmtTime(v, 0)} {...AXIS} />
        <YAxis {...AXIS} domain={["auto", "auto"]} tickFormatter={(v: number) => fmtNum(v, 0)} />
        <Tooltip contentStyle={tooltipStyle} labelFormatter={(v) => fmtTime(Number(v))} formatter={(v, n) => [`${fmtNum(Number(v), 2)} ${unit}`, String(n)]} />
        <Legend wrapperStyle={{ fontSize: 11 }} />
        <Line name="raw" dataKey="raw" stroke="#8a9bb8" strokeWidth={0.8} strokeOpacity={0.7} dot={false} isAnimationActive={false} connectNulls={false} />
        <Line name="smoothed" dataKey="smooth" stroke="#22d3ee" strokeWidth={2} dot={false} isAnimationActive={false} connectNulls={false} />
      </LineChart>
    </ResponsiveContainer>
  );
}

export function PosePanel({ sessionId, onSeek }: { sessionId: string; onSeek: (t: number) => void }) {
  const [landmark, setLandmark] = useState("right_wrist");
  const [axis, setAxis] = useState<"y" | "x">("y");
  const [angle, setAngle] = useState(ANGLES[0].key);
  const [windowS, setWindowS] = useState(0.3);
  const ps = useAsync(() => poseApi.series(sessionId, landmark, windowS), [sessionId, landmark, windowS]);
  const angleDef = ANGLES.find((a) => a.key === angle)!;
  const selectClass = "rounded-md border border-edge bg-navy-850 px-2 py-1 text-xs text-soft outline-none focus:border-accent";
  return (
    <Card title="Pose signals" subtitle="2-D monocular proxies, not clinical joint angles — raw landmarks (thin) versus temporally smoothed (solid). Click a chart to seek.">
      <div className="mb-4 flex flex-wrap items-center gap-3 text-xs text-muted">
        <label className="flex items-center gap-2">
          Landmark
          <select className={selectClass} value={landmark} onChange={(e) => setLandmark(e.target.value)}>
            {LANDMARKS.map((l) => (
              <option key={l} value={l}>
                {titleCase(l)}
              </option>
            ))}
          </select>
        </label>
        <Segmented value={axis} onChange={setAxis} options={[{ value: "y", label: "Vertical (y)" }, { value: "x", label: "Horizontal (x)" }]} />
        <label className="flex items-center gap-2">
          Angle proxy
          <select className={selectClass} value={angle} onChange={(e) => setAngle(e.target.value)}>
            {ANGLES.map((a) => (
              <option key={a.key} value={a.key}>
                {a.label}
              </option>
            ))}
          </select>
        </label>
        <label className="flex items-center gap-2">
          Smoothing window
          <input type="range" min={0.1} max={1} step={0.05} value={windowS} onChange={(e) => setWindowS(Number(e.target.value))} />
          <span className="tabular text-soft">{windowS.toFixed(2)} s</span>
        </label>
      </div>
      {ps.error ? (
        <ErrorBox error={ps.error} />
      ) : !ps.data ? (
        <Skeleton className="h-[460px]" />
      ) : (
        <div className="grid gap-5 xl:grid-cols-2">
          <div>
            <div className="mb-1 text-xs font-medium text-soft">
              {titleCase(landmark)} · {axis === "y" ? "image y (px, down is positive)" : "image x (px)"}
            </div>
            <RawSmoothChart t={ps.data.t} raw={axis === "y" ? ps.data.y_raw : ps.data.x_raw} smooth={axis === "y" ? ps.data.y_smooth : ps.data.x_smooth} unit="px" onSeek={onSeek} />
          </div>
          <div>
            <div className="mb-1 text-xs font-medium text-soft">
              {angleDef.label} ({angleDef.unit})
            </div>
            <RawSmoothChart t={ps.data.t} raw={series(ps.data, `${angle}_raw`)} smooth={series(ps.data, `${angle}_smooth`)} unit={angleDef.unit} onSeek={onSeek} />
          </div>
        </div>
      )}
      <div className="mt-3">
        <Notice tone="info">
          Angles are measured in the image plane of one camera, so they change with viewpoint and are only meaningful for comparing
          moments within the same session. They are not anatomical joint angles and must not be used for injury or medical judgements.
        </Notice>
      </div>
    </Card>
  );
}
