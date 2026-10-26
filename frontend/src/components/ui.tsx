import { AlertTriangle, FlaskConical, Info, Loader2 } from "lucide-react";
import { useState, type ButtonHTMLAttributes, type ReactNode } from "react";

export function cx(...parts: (string | false | null | undefined)[]): string {
  return parts.filter(Boolean).join(" ");
}

export function Card({ children, className, title, actions, subtitle }: {
  children: ReactNode;
  className?: string;
  title?: ReactNode;
  subtitle?: ReactNode;
  actions?: ReactNode;
}) {
  return (
    <section className={cx("rounded-2xl border border-edge bg-navy-800/80 shadow-[0_1px_0_rgba(255,255,255,0.03)_inset] backdrop-blur", className)}>
      {(title || actions) && (
        <header className="flex items-start justify-between gap-3 border-b border-edge/70 px-5 py-3.5">
          <div>
            {title && <h2 className="text-sm font-semibold tracking-wide text-white">{title}</h2>}
            {subtitle && <p className="mt-0.5 text-xs text-muted">{subtitle}</p>}
          </div>
          {actions && <div className="flex flex-wrap items-center gap-2">{actions}</div>}
        </header>
      )}
      <div className="p-5">{children}</div>
    </section>
  );
}

type Variant = "primary" | "ghost" | "danger" | "subtle";

export function Button({ variant = "subtle", loading, className, children, ...rest }: ButtonHTMLAttributes<HTMLButtonElement> & {
  variant?: Variant;
  loading?: boolean;
}) {
  const styles: Record<Variant, string> = {
    primary: "bg-accent text-navy-900 hover:bg-cyan-300 shadow-[0_0_24px_-6px_rgba(34,211,238,0.6)]",
    subtle: "bg-navy-700 text-soft hover:bg-navy-600 border border-edge",
    ghost: "text-soft hover:bg-navy-700",
    danger: "bg-red-500/10 text-bad border border-red-500/30 hover:bg-red-500/20",
  };
  return (
    <button
      {...rest}
      disabled={rest.disabled || loading}
      className={cx(
        "inline-flex items-center justify-center gap-2 rounded-lg px-3.5 py-2 text-sm font-medium transition duration-150 disabled:cursor-not-allowed disabled:opacity-50",
        styles[variant],
        className,
      )}
    >
      {loading && <Loader2 className="h-4 w-4 animate-spin" />}
      {children}
    </button>
  );
}

export function Badge({ children, tone = "neutral", className }: {
  children: ReactNode;
  tone?: "neutral" | "accent" | "good" | "warn" | "bad" | "synthetic";
  className?: string;
}) {
  const tones = {
    neutral: "bg-navy-700 text-soft border-edge",
    accent: "bg-cyan-400/10 text-accent border-cyan-400/30",
    good: "bg-lime-400/10 text-good border-lime-400/30",
    warn: "bg-amber-400/10 text-warn border-amber-400/30",
    bad: "bg-red-400/10 text-bad border-red-400/30",
    synthetic: "bg-orange-400/15 text-synthetic border-orange-400/40",
  };
  return (
    <span className={cx("inline-flex items-center gap-1 rounded-full border px-2 py-0.5 text-[11px] font-semibold uppercase tracking-wider", tones[tone], className)}>
      {children}
    </span>
  );
}

export function SyntheticBadge() {
  return (
    <Badge tone="synthetic">
      <FlaskConical className="h-3 w-3" /> Synthetic demo data
    </Badge>
  );
}

export function SyntheticBanner() {
  return (
    <div className="flex items-start gap-3 rounded-xl border border-orange-400/40 bg-orange-400/10 px-4 py-3 text-sm text-orange-200">
      <FlaskConical className="mt-0.5 h-4 w-4 shrink-0 text-synthetic" />
      <div>
        <span className="font-semibold uppercase tracking-wider text-synthetic">Synthetic demo data</span> — this session was
        rendered by the built-in simulator and analysed with ground-truth detections. Its numbers describe a simulation, not a
        real player's performance.
      </div>
    </div>
  );
}

export function ExperimentalBadge({ label = "Experimental" }: { label?: string }) {
  return (
    <Badge tone="warn">
      <FlaskConical className="h-3 w-3" /> {label}
    </Badge>
  );
}

export function InfoTip({ text, className }: { text: ReactNode; className?: string }) {
  const [open, setOpen] = useState(false);
  return (
    <span
      className={cx("relative inline-flex", className)}
      onMouseEnter={() => setOpen(true)}
      onMouseLeave={() => setOpen(false)}
      onFocus={() => setOpen(true)}
      onBlur={() => setOpen(false)}
      tabIndex={0}
      aria-label="More information"
    >
      <Info className="h-3.5 w-3.5 cursor-help text-muted hover:text-accent" />
      {open && (
        <span className="absolute bottom-full left-1/2 z-50 mb-2 w-72 -translate-x-1/2 rounded-lg border border-edge bg-ink px-3 py-2 text-xs font-normal normal-case leading-relaxed tracking-normal text-soft shadow-2xl">
          {text}
        </span>
      )}
    </span>
  );
}

export function Spinner({ label }: { label?: string }) {
  return (
    <div className="flex items-center gap-2 text-sm text-muted">
      <Loader2 className="h-4 w-4 animate-spin text-accent" /> {label ?? "Loading…"}
    </div>
  );
}

export function ErrorBox({ error, action }: { error: unknown; action?: ReactNode }) {
  const msg = error instanceof Error ? error.message : String(error);
  return (
    <div className="flex items-start gap-3 rounded-xl border border-red-500/30 bg-red-500/10 px-4 py-3 text-sm text-red-200">
      <AlertTriangle className="mt-0.5 h-4 w-4 shrink-0 text-bad" />
      <div className="flex-1">{msg}</div>
      {action}
    </div>
  );
}

export function Notice({ children, tone = "warn" }: { children: ReactNode; tone?: "warn" | "info" }) {
  return (
    <div
      className={cx(
        "flex items-start gap-3 rounded-xl border px-4 py-3 text-sm",
        tone === "warn" ? "border-amber-400/30 bg-amber-400/10 text-amber-100" : "border-cyan-400/25 bg-cyan-400/5 text-cyan-50",
      )}
    >
      {tone === "warn" ? <AlertTriangle className="mt-0.5 h-4 w-4 shrink-0 text-warn" /> : <Info className="mt-0.5 h-4 w-4 shrink-0 text-accent" />}
      <div className="flex-1 leading-relaxed">{children}</div>
    </div>
  );
}

export function EmptyState({ icon, title, children }: { icon?: ReactNode; title: string; children?: ReactNode }) {
  return (
    <div className="flex flex-col items-center justify-center gap-3 rounded-2xl border border-dashed border-edge px-6 py-14 text-center">
      {icon && <div className="text-accent">{icon}</div>}
      <h3 className="text-base font-semibold text-white">{title}</h3>
      {children && <div className="max-w-md text-sm text-muted">{children}</div>}
    </div>
  );
}

export function Stat({ label, value, unit, hint, tone, sub }: {
  label: string;
  value: ReactNode;
  unit?: string;
  hint?: ReactNode;
  tone?: "good" | "accent";
  sub?: ReactNode;
}) {
  return (
    <div className="rounded-xl border border-edge bg-navy-850 px-4 py-3.5 transition hover:border-cyan-400/30">
      <div className="flex items-center gap-1.5 text-[11px] font-medium uppercase tracking-wider text-muted">
        {label}
        {hint && <InfoTip text={hint} />}
      </div>
      <div className={cx("mt-1.5 text-2xl font-semibold tabular", tone === "good" ? "text-good" : tone === "accent" ? "text-accent" : "text-white")}>
        {value}
        {unit && <span className="ml-1 text-sm font-medium text-muted">{unit}</span>}
      </div>
      {sub && <div className="mt-0.5 text-xs text-muted">{sub}</div>}
    </div>
  );
}

export function Toggle({ checked, onChange, label }: { checked: boolean; onChange: (v: boolean) => void; label: ReactNode }) {
  return (
    <label className="inline-flex cursor-pointer select-none items-center gap-2 text-xs text-soft">
      <span
        role="switch"
        aria-checked={checked}
        tabIndex={0}
        onKeyDown={(e) => (e.key === " " || e.key === "Enter") && onChange(!checked)}
        onClick={() => onChange(!checked)}
        className={cx("relative h-4.5 w-8 rounded-full transition", checked ? "bg-accent" : "bg-navy-600")}
      >
        <span className={cx("absolute top-0.5 h-3.5 w-3.5 rounded-full bg-white transition", checked ? "left-4" : "left-0.5")} />
      </span>
      {label}
    </label>
  );
}

export function Segmented<T extends string>({ value, options, onChange }: {
  value: T;
  options: { value: T; label: ReactNode; title?: string }[];
  onChange: (v: T) => void;
}) {
  return (
    <div className="inline-flex rounded-lg border border-edge bg-navy-850 p-0.5">
      {options.map((o) => (
        <button
          key={o.value}
          title={o.title}
          onClick={() => onChange(o.value)}
          className={cx(
            "rounded-md px-2.5 py-1 text-xs font-medium transition",
            value === o.value ? "bg-navy-600 text-accent shadow" : "text-muted hover:text-soft",
          )}
        >
          {o.label}
        </button>
      ))}
    </div>
  );
}

export function Field({ label, children, hint }: { label: string; children: ReactNode; hint?: ReactNode }) {
  return (
    <label className="block">
      <span className="mb-1.5 flex items-center gap-1.5 text-xs font-medium text-muted">
        {label}
        {hint && <InfoTip text={hint} />}
      </span>
      {children}
    </label>
  );
}

export const inputClass =
  "w-full rounded-lg border border-edge bg-navy-850 px-3 py-2 text-sm text-soft outline-none transition focus:border-accent focus:ring-2 focus:ring-cyan-400/20";

export function Skeleton({ className }: { className?: string }) {
  return <div className={cx("skeleton", className)} />;
}
