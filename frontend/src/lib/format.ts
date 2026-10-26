export function isNum(v: unknown): v is number {
  return typeof v === "number" && Number.isFinite(v);
}

export function fmtNum(v: unknown, digits = 1, unit = ""): string {
  if (!isNum(v)) return "—";
  const s = v.toLocaleString("en-US", { minimumFractionDigits: digits, maximumFractionDigits: digits });
  return unit ? `${s} ${unit}` : s;
}

export function fmtPct(fraction: unknown, digits = 0): string {
  return isNum(fraction) ? `${(fraction * 100).toFixed(digits)}%` : "—";
}

/** m:ss */
export function fmtDuration(seconds: unknown): string {
  if (!isNum(seconds) || seconds < 0) return "—";
  const m = Math.floor(seconds / 60);
  const s = Math.floor(seconds % 60);
  return `${m}:${s.toString().padStart(2, "0")}`;
}

/** m:ss.d */
export function fmtTime(seconds: unknown, digits = 1): string {
  if (!isNum(seconds) || seconds < 0) return "—";
  const m = Math.floor(seconds / 60);
  const s = seconds - m * 60;
  const width = digits > 0 ? digits + 3 : 2;
  return `${m}:${s.toFixed(digits).padStart(width, "0")}`;
}

export function fmtBytes(n: unknown): string {
  if (!isNum(n)) return "—";
  const units = ["B", "KB", "MB", "GB"];
  let v = n;
  let i = 0;
  while (v >= 1024 && i < units.length - 1) {
    v /= 1024;
    i++;
  }
  return `${v.toFixed(i ? 1 : 0)} ${units[i]}`;
}

export function fmtSigned(v: unknown, digits = 2): string {
  if (!isNum(v)) return "—";
  const s = v.toFixed(digits);
  return v > 0 ? `+${s}` : s;
}

export function titleCase(s: string): string {
  return s.replace(/_/g, " ").replace(/\b\w/g, (c) => c.toUpperCase());
}

export function relativeDate(iso: string): string {
  const d = new Date(iso);
  if (Number.isNaN(d.getTime())) return "";
  const diff = (Date.now() - d.getTime()) / 1000;
  if (diff < 60) return "just now";
  if (diff < 3600) return `${Math.floor(diff / 60)} min ago`;
  if (diff < 86400) return `${Math.floor(diff / 3600)} h ago`;
  return d.toLocaleDateString();
}
