import { FileDown } from "lucide-react";
import { useState } from "react";
import { useAsync } from "../hooks/useAsync";
import { isNum, titleCase } from "../lib/format";
import { api, ApiError, poseApi, saveBlob } from "../services/api";
import { Button, Card, ErrorBox, Notice, Spinner, SyntheticBadge } from "./ui";

function fmt(v: unknown): string {
  if (isNum(v)) return Number.isInteger(v) ? String(v) : Math.abs(v) < 10 ? v.toFixed(3) : v.toFixed(1);
  if (v === null || v === undefined) return "—";
  if (typeof v === "boolean") return v ? "yes" : "no";
  if (Array.isArray(v)) return v.length > 6 ? `${v.length} items` : v.map((x) => (typeof x === "object" ? JSON.stringify(x) : String(x))).join(", ");
  return String(v);
}

function Section({ title, data }: { title: string; data: Record<string, unknown> }) {
  const flat: [string, unknown][] = [];
  for (const [k, v] of Object.entries(data)) {
    if (v && typeof v === "object" && !Array.isArray(v) && k !== "confusion_matrix") {
      for (const [k2, v2] of Object.entries(v as Record<string, unknown>)) if (typeof v2 !== "object" || v2 === null) flat.push([`${k} · ${k2}`, v2]);
    } else if (k !== "confusion_matrix") flat.push([k, v]);
  }
  return (
    <div className="rounded-xl border border-edge bg-navy-850 p-4">
      <h3 className="mb-2 text-sm font-semibold text-accent">{titleCase(title)}</h3>
      <dl className="space-y-1 text-xs">
        {flat.map(([k, v]) => (
          <div key={k} className="flex justify-between gap-3 border-b border-edge/40 pb-1 last:border-0">
            <dt className="text-muted">{titleCase(k)}</dt>
            <dd className="max-w-[60%] truncate text-right tabular text-soft" title={fmt(v)}>
              {fmt(v)}
            </dd>
          </div>
        ))}
      </dl>
    </div>
  );
}

function TemplateDownload({ sessionId }: { sessionId: string }) {
  const [busy, setBusy] = useState(false);
  const [warning, setWarning] = useState<string | null>(null);
  const [error, setError] = useState<unknown>(null);
  return (
    <div className="space-y-2">
      <Button
        loading={busy}
        onClick={async () => {
          setBusy(true);
          setError(null);
          try {
            const doc = await poseApi.template(sessionId);
            const w = doc.warning;
            setWarning(typeof w === "string" ? w : null);
            saveBlob(new Blob([JSON.stringify(doc, null, 1)], { type: "application/json" }), `annotation_template_${sessionId.slice(0, 8)}.json`);
          } catch (e) {
            setError(e);
          } finally {
            setBusy(false);
          }
        }}
      >
        <FileDown className="h-4 w-4" /> Download annotation template
      </Button>
      {warning && <Notice>{warning}</Notice>}
      {error ? <ErrorBox error={error} /> : null}
    </div>
  );
}

export function EvaluationPanel({ sessionId }: { sessionId: string }) {
  const ev = useAsync(() => api.evaluation(sessionId), [sessionId]);
  if (ev.loading) return <Spinner label="Evaluating against annotations…" />;
  if (ev.error)
    return (
      <div className="space-y-3">
        <Notice tone="info">
          {ev.error instanceof ApiError && ev.error.status === 409
            ? "No benchmark annotations for this session yet. Download the machine-generated template, correct it by hand (see docs/annotation.md), and save it to measure detection, tracking, position and stroke accuracy."
            : String(ev.error instanceof Error ? ev.error.message : ev.error)}
        </Notice>
        <TemplateDownload sessionId={sessionId} />
      </div>
    );
  const d = ev.data ?? {};
  const synthetic = d.source === "synthetic";
  const sections = Object.entries(d).filter(([, v]) => v && typeof v === "object" && !Array.isArray(v));
  return (
    <Card
      title="Evaluation against ground truth"
      subtitle={synthetic ? "Ground truth from the simulator — measures pipeline mechanics, not real-world accuracy." : "Measured against your manual annotations."}
      actions={synthetic ? <SyntheticBadge /> : undefined}
    >
      <div className="mb-4">
        <TemplateDownload sessionId={sessionId} />
      </div>
      <div className="grid gap-4 md:grid-cols-2 xl:grid-cols-3">
        {sections.map(([k, v]) => (
          <Section key={k} title={k} data={v as Record<string, unknown>} />
        ))}
      </div>
    </Card>
  );
}
