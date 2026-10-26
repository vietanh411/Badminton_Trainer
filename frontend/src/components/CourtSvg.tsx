import { useMemo, type ReactNode } from "react";
import { gridToRgba } from "../lib/colormap";
import type { CourtGeometry, HeatmapPayload } from "../types/api";

export const COURT_MARGIN = 1.0;

/** Court metres → SVG user units (far baseline at the top). */
export function toSvg(court: CourtGeometry, x: number, y: number): [number, number] {
  return [x, court.length_m - y];
}

export function CourtSvg({
  court,
  children,
  className,
  margin = COURT_MARGIN,
  below,
  showZones = false,
}: {
  court: CourtGeometry;
  children?: ReactNode;
  className?: string;
  margin?: number;
  below?: ReactNode;
  showZones?: boolean;
}) {
  const W = court.width_m;
  const L = court.length_m;
  const alley = (W - court.singles_width_m) / 2;
  return (
    <svg viewBox={`${-margin} ${-margin} ${W + 2 * margin} ${L + 2 * margin}`} className={className} role="img" aria-label="Top-down badminton court">
      <defs>
        <linearGradient id="courtFloor" x1="0" y1="0" x2="0" y2="1">
          <stop offset="0" stopColor="#0f1d33" />
          <stop offset="1" stopColor="#0c1628" />
        </linearGradient>
      </defs>
      <rect x={-margin} y={-margin} width={W + 2 * margin} height={L + 2 * margin} fill="#0a1222" rx={0.3} />
      <rect x={0} y={0} width={W} height={L} fill="url(#courtFloor)" />
      {below}
      {showZones && <ZoneGuides court={court} alley={alley} />}
      {court.lines.map((ln) => {
        const [x1, y1] = toSvg(court, ln.start[0], ln.start[1]);
        const [x2, y2] = toSvg(court, ln.end[0], ln.end[1]);
        const net = ln.kind === "net";
        return (
          <line
            key={ln.id}
            x1={x1}
            y1={y1}
            x2={x2}
            y2={y2}
            stroke={net ? "#e2e8f0" : "#c9d4e5"}
            strokeOpacity={net ? 0.9 : 0.75}
            strokeWidth={net ? 0.07 : 0.045}
            strokeDasharray={net ? "0.18 0.12" : undefined}
          >
            <title>{ln.label}</title>
          </line>
        );
      })}
      {children}
    </svg>
  );
}

/** Faint front/mid/rear and left/centre/right guides for both singles halves. */
function ZoneGuides({ court, alley }: { court: CourtGeometry; alley: number }) {
  const L = court.length_m;
  const net = court.net_y_m;
  const ss = 1.98;
  const mid = (ss + net) / 2;
  const third = court.singles_width_m / 3;
  const ys = [net - ss, net - mid, net + ss, net + mid];
  return (
    <g stroke="#22d3ee" strokeOpacity={0.18} strokeWidth={0.025} strokeDasharray="0.1 0.1">
      {ys.map((y) => (
        <line key={y} x1={alley} x2={court.width_m - alley} y1={L - y} y2={L - y} />
      ))}
      {[alley + third, alley + 2 * third].map((x) => (
        <line key={x} x1={x} x2={x} y1={0} y2={L} />
      ))}
    </g>
  );
}

/** Heatmap grid rendered as an <image> in court coordinates (drawn beneath the court lines). */
export function HeatmapLayer({ court, hm, opacity, vmax }: { court: CourtGeometry; hm: HeatmapPayload; opacity: number; vmax?: number }) {
  const href = useMemo(() => {
    if (!hm.values || !hm.nx || !hm.ny) return null;
    const canvas = document.createElement("canvas");
    canvas.width = hm.nx;
    canvas.height = hm.ny;
    const ctx = canvas.getContext("2d");
    if (!ctx) return null;
    const px = gridToRgba(hm.values, vmax ?? hm.max ?? 0, opacity);
    ctx.putImageData(new ImageData(new Uint8ClampedArray(px), hm.nx, hm.ny), 0, 0);
    return canvas.toDataURL();
  }, [hm, opacity, vmax]);
  if (!href || hm.x0 === undefined || hm.y0 === undefined || !hm.cell_m || !hm.nx || !hm.ny) return null;
  const w = hm.nx * hm.cell_m;
  const h = hm.ny * hm.cell_m;
  return <image href={href} x={hm.x0} y={court.length_m - (hm.y0 + h)} width={w} height={h} preserveAspectRatio="none" />;
}
