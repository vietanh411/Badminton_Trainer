import { Download, FileJson, FileSpreadsheet, FileText, Image } from "lucide-react";
import type { ReactNode } from "react";
import { mediaUrl } from "../services/api";
import { Card, SyntheticBadge } from "./ui";

const ITEMS: { path: string; label: string; desc: string; icon: ReactNode }[] = [
  { path: "report", label: "PDF analysis report", desc: "Summary, heatmap, speed and zones, stroke list, data quality and caveats.", icon: <FileText className="h-5 w-5" /> },
  { path: "export/positions.csv", label: "Positions (CSV)", desc: "Per-frame raw and filtered court coordinates, source, confidence, speed, zone.", icon: <FileSpreadsheet className="h-5 w-5" /> },
  { path: "export/events.csv", label: "Events (CSV)", desc: "Direction changes, high-speed runs, excursions, tracking gaps, identity links.", icon: <FileSpreadsheet className="h-5 w-5" /> },
  { path: "export/strokes.csv", label: "Strokes (CSV)", desc: "Detected and manual stroke events with review status and your labels.", icon: <FileSpreadsheet className="h-5 w-5" /> },
  { path: "export/metrics.json", label: "Metrics (JSON)", desc: "All movement, zone, recovery and quality metrics.", icon: <FileJson className="h-5 w-5" /> },
  { path: "export/analysis.json", label: "Full analysis (JSON)", desc: "Metrics, settings, timings, calibration, events and strokes in one file.", icon: <FileJson className="h-5 w-5" /> },
  { path: "heatmap?format=png&trajectory=true", label: "Heatmap (PNG)", desc: "Whole-session occupancy heatmap with trajectory.", icon: <Image className="h-5 w-5" /> },
  { path: "heatmap?format=svg", label: "Heatmap (SVG)", desc: "Vector version for reports and slides.", icon: <Image className="h-5 w-5" /> },
];

export function ExportPanel({ sessionId, synthetic }: { sessionId: string; synthetic: boolean }) {
  const href = (p: string) => {
    const [path, query] = p.split("?");
    const params = Object.fromEntries(new URLSearchParams(query ?? ""));
    return mediaUrl(`/api/sessions/${sessionId}/${path}`, params);
  };
  return (
    <Card title="Export" subtitle="Downloads are generated from the latest completed analysis" actions={synthetic ? <SyntheticBadge /> : undefined}>
      <div className="grid gap-3 sm:grid-cols-2 xl:grid-cols-4">
        {ITEMS.map((it) => (
          <a
            key={it.path}
            href={href(it.path)}
            className="group flex flex-col gap-2 rounded-xl border border-edge bg-navy-850 p-4 transition hover:-translate-y-0.5 hover:border-cyan-400/50"
          >
            <div className="flex items-center justify-between text-accent">
              {it.icon}
              <Download className="h-4 w-4 text-muted transition group-hover:text-accent" />
            </div>
            <div className="text-sm font-semibold text-white">{it.label}</div>
            <div className="text-xs text-muted">{it.desc}</div>
          </a>
        ))}
      </div>
      {synthetic && <p className="mt-3 text-xs text-synthetic">Exports of synthetic sessions are marked as synthetic demo data inside the files.</p>}
    </Card>
  );
}
