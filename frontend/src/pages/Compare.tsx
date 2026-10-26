import { AlertTriangle, ArrowDownRight, ArrowUpRight, GitCompare, Minus } from "lucide-react";
import { useEffect, useMemo, useState } from "react";
import { useSearchParams } from "react-router-dom";
import { CourtSvg, HeatmapLayer } from "../components/CourtSvg";
import { Card, cx, EmptyState, ErrorBox, inputClass, Skeleton, Spinner, SyntheticBadge } from "../components/ui";
import { useAsync } from "../hooks/useAsync";
import { useCourt } from "../hooks/useStatic";
import { cssGradient } from "../lib/colormap";
import { fmtNum, fmtSigned, isNum } from "../lib/format";
import { api } from "../services/api";
import type { CompareRow, HeatmapPayload, Session } from "../types/api";

function fmtVal(v: unknown, unit: string): string {
  if (!isNum(v)) return v === null || v === undefined ? "—" : String(v);
  if (unit === "fraction") return `${(v * 100).toFixed(1)}%`;
  if (unit === "" && Number.isInteger(v)) return String(v); // counts in the data-quality table
  return fmtNum(v, Math.abs(v) >= 100 ? 0 : 2);
}

function DiffCell({ r }: { r: CompareRow }) {
  if (!isNum(r.diff)) return <span className="text-muted">—</span>;
  const better = r.direction === "neutral" || r.diff === 0 ? null : (r.diff > 0) === (r.direction === "more");
  const Icon = r.diff === 0 ? Minus : r.diff > 0 ? ArrowUpRight : ArrowDownRight;
  return (
    <span className={cx("inline-flex items-center gap-1 tabular", better === null ? "text-soft" : better ? "text-good" : "text-warn")}>
      <Icon className="h-3.5 w-3.5" />
      {r.unit === "fraction" ? `${fmtSigned(r.diff * 100, 1)} pp` : fmtSigned(r.diff, 2)}
      {isNum(r.rel_pct) && r.unit !== "fraction" && <span className="text-muted">({fmtSigned(r.rel_pct, 0)}%)</span>}
    </span>
  );
}

function SessionPicker({ value, onChange, sessions, label }: { value: string; onChange: (v: string) => void; sessions: Session[]; label: string }) {
  return (
    <label className="block flex-1">
      <span className="mb-1 block text-xs font-medium text-muted">{label}</span>
      <select className={inputClass} value={value} onChange={(e) => onChange(e.target.value)}>
        <option value="">Choose a session…</option>
        {sessions.map((s) => (
          <option key={s.id} value={s.id}>
            {s.name}
            {s.data_source === "synthetic" ? " (synthetic)" : ""}
          </option>
        ))}
      </select>
    </label>
  );
}

export function ComparePage() {
  const court = useCourt();
  const [params, setParams] = useSearchParams();
  const sessions = useAsync(() => api.sessions(), []);
  const done = useMemo(() => (sessions.data ?? []).filter((s) => s.status === "completed" && s.summary?.calibrated !== false), [sessions.data]);
  const [a, setA] = useState(params.get("a") ?? "");
  const [b, setB] = useState(params.get("b") ?? "");

  useEffect(() => {
    if (!a && !b && done.length >= 2) {
      setA(done[1].id);
      setB(done[0].id);
    }
  }, [done, a, b]);
  useEffect(() => {
    if (a && b) setParams({ a, b }, { replace: true });
  }, [a, b, setParams]);

  const ready = !!a && !!b && a !== b;
  const cmp = useAsync(() => api.compare(a, b), [a, b], ready);
  const hmA = useAsync(() => api.heatmap(a, { normalize: "share" }), [a], !!a);
  const hmB = useAsync(() => api.heatmap(b, { normalize: "share" }), [b], !!b);
  const vmax = Math.max(hmA.data?.max ?? 0, hmB.data?.max ?? 0);
  const name = (id: string) => done.find((s) => s.id === id)?.name ?? "";
  const syn = (id: string) => done.find((s) => s.id === id)?.data_source === "synthetic";

  if (sessions.loading && !sessions.data) return <Spinner />;
  if (done.length < 2)
    return (
      <EmptyState icon={<GitCompare className="h-10 w-10" />} title="Two analysed sessions needed">
        Analyse at least two calibrated sessions (or create both demo sessions from the dashboard) to compare them.
      </EmptyState>
    );

  const heat = (hm: HeatmapPayload | undefined, loading: boolean, id: string) => (
    <div className="space-y-2">
      <div className="min-h-12 space-y-1">
        <div className="truncate text-sm font-semibold text-white" title={name(id)}>
          {name(id)}
        </div>
        {syn(id) && <SyntheticBadge />}
      </div>
      {court && hm?.available ? (
        <CourtSvg court={court} className="w-full" below={<HeatmapLayer court={court} hm={hm} opacity={0.9} vmax={vmax} />} />
      ) : loading ? (
        <Skeleton className="aspect-[8.1/15.4] w-full" />
      ) : (
        <div className="text-xs text-muted">{hm?.reason ?? "No heatmap"}</div>
      )}
    </div>
  );

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-bold tracking-tight text-white">Session comparison</h1>
        <p className="mt-1 text-sm text-muted">Normalised metrics (per minute, shares of time) so sessions of different length and frame rate compare fairly.</p>
      </div>
      <Card>
        <div className="flex flex-col gap-4 md:flex-row">
          <SessionPicker label="Session A (baseline)" value={a} onChange={setA} sessions={done} />
          <SessionPicker label="Session B" value={b} onChange={setB} sessions={done} />
        </div>
        {a && a === b && <div className="mt-2 text-xs text-warn">Choose two different sessions.</div>}
      </Card>
      {ready && (
        <div className="grid gap-5 xl:grid-cols-[minmax(0,560px)_minmax(0,1fr)]">
          <Card title="Occupancy heatmaps" subtitle="Share of observed time · same colour scale for both">
            <div className="grid grid-cols-2 gap-4">
              {heat(hmA.data, hmA.loading, a)}
              {heat(hmB.data, hmB.loading, b)}
            </div>
            <div className="mt-3">
              <div className="h-2 rounded-full" style={{ background: cssGradient() }} />
              <div className="mt-1 flex justify-between text-[11px] tabular text-muted">
                <span>0%</span>
                <span>{fmtNum(vmax * 100, 2)}% of time per cell</span>
              </div>
            </div>
          </Card>
          <div className="space-y-5">
            {cmp.error ? <ErrorBox error={cmp.error} /> : null}
            {cmp.data?.warnings.map((w) => (
              <div key={w} className="flex items-start gap-2 rounded-xl border border-amber-400/30 bg-amber-400/10 px-4 py-2.5 text-sm text-amber-100">
                <AlertTriangle className="mt-0.5 h-4 w-4 shrink-0 text-warn" /> {w}
              </div>
            ))}
            <Card title="Metrics" subtitle="B − A. Green/amber reflect the usual direction of improvement; context matters.">
              {!cmp.data ? (
                <Skeleton className="h-80" />
              ) : (
                <table className="w-full text-sm">
                  <thead className="text-left text-xs text-muted">
                    <tr className="border-b border-edge">
                      <th className="py-2 font-medium">Metric</th>
                      <th className="pl-4 text-right font-medium">A</th>
                      <th className="pl-4 text-right font-medium">B</th>
                      <th className="text-right font-medium">Difference</th>
                    </tr>
                  </thead>
                  <tbody>
                    {cmp.data.metrics.map((r) => (
                      <tr key={r.key} className="border-b border-edge/50">
                        <td className="py-2 text-soft">
                          {r.label}
                          {r.unit !== "fraction" && <span className="ml-1 text-xs text-muted">{r.unit}</span>}
                          {r.key === "composite" && <span className="ml-1 text-[10px] uppercase text-warn">experimental</span>}
                        </td>
                        <td className="pl-4 text-right tabular">{fmtVal(r.a, r.unit)}</td>
                        <td className="pl-4 text-right tabular">{fmtVal(r.b, r.unit)}</td>
                        <td className="pl-4 text-right whitespace-nowrap">
                          <DiffCell r={r} />
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              )}
            </Card>
            {cmp.data && (
              <Card title="Data quality" subtitle="Differences here can explain metric differences">
                <table className="w-full text-sm">
                  <tbody>
                    {cmp.data.quality.map((q) => (
                      <tr key={q.key} className="border-b border-edge/50 last:border-0">
                        <td className="py-1.5 text-muted">{q.label}</td>
                        <td className="pl-4 text-right tabular">{fmtVal(q.a, "")}</td>
                        <td className="pl-4 text-right tabular">{fmtVal(q.b, "")}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </Card>
            )}
          </div>
        </div>
      )}
    </div>
  );
}
