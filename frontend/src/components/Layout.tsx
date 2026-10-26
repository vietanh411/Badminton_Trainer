import { Activity, GitCompare, KeyRound, LayoutDashboard, Upload, X } from "lucide-react";
import { useState, type ReactNode } from "react";
import { NavLink, Outlet } from "react-router-dom";
import { useSystem } from "../hooks/useStatic";
import { getApiKey, setApiKey } from "../services/api";
import { Button, cx, inputClass } from "./ui";

function NavItem({ to, icon, children, end }: { to: string; icon: ReactNode; children: ReactNode; end?: boolean }) {
  return (
    <NavLink
      to={to}
      end={end}
      className={({ isActive }) =>
        cx(
          "flex items-center gap-3 rounded-lg px-3 py-2 text-sm font-medium transition",
          isActive ? "bg-cyan-400/10 text-accent" : "text-muted hover:bg-navy-700 hover:text-soft",
        )
      }
    >
      {icon}
      {children}
    </NavLink>
  );
}

function SettingsDialog({ onClose }: { onClose: () => void }) {
  const [key, setKey] = useState(getApiKey() ?? "");
  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 p-4" onClick={onClose}>
      <div className="w-full max-w-md rounded-2xl border border-edge bg-navy-800 p-6 fade-in" onClick={(e) => e.stopPropagation()}>
        <div className="mb-4 flex items-center justify-between">
          <h2 className="font-semibold text-white">API access</h2>
          <button onClick={onClose} className="text-muted hover:text-soft" aria-label="Close">
            <X className="h-4 w-4" />
          </button>
        </div>
        <p className="mb-3 text-sm text-muted">
          Only needed when the backend runs with <code className="text-accent">BV_API_KEY</code>. The key is kept in this
          browser's local storage.
        </p>
        <input className={inputClass} type="password" value={key} placeholder="API key" onChange={(e) => setKey(e.target.value)} />
        <div className="mt-4 flex justify-end gap-2">
          <Button
            variant="ghost"
            onClick={() => {
              setApiKey(null);
              onClose();
              window.location.reload();
            }}
          >
            Clear
          </Button>
          <Button
            variant="primary"
            onClick={() => {
              setApiKey(key.trim() || null);
              onClose();
              window.location.reload();
            }}
          >
            Save
          </Button>
        </div>
      </div>
    </div>
  );
}

export function Layout() {
  const sys = useSystem();
  const [settings, setSettings] = useState(false);
  return (
    <div className="flex min-h-screen">
      <aside className="sticky top-0 hidden h-screen w-60 shrink-0 flex-col border-r border-edge bg-ink/60 px-4 py-5 backdrop-blur md:flex">
        <NavLink to="/" className="mb-8 flex items-center gap-2.5 px-2">
          <span className="grid h-9 w-9 place-items-center rounded-xl bg-gradient-to-br from-cyan-400/30 to-lime-400/10 ring-1 ring-cyan-400/40">
            <Activity className="h-5 w-5 text-accent" />
          </span>
          <span>
            <span className="block text-sm font-bold tracking-wide text-white">BadmintonVision</span>
            <span className="block text-[10px] font-semibold uppercase tracking-[0.2em] text-accent">AI analytics</span>
          </span>
        </NavLink>
        <nav className="flex flex-col gap-1">
          <NavItem to="/" end icon={<LayoutDashboard className="h-4 w-4" />}>
            Dashboard
          </NavItem>
          <NavItem to="/new" icon={<Upload className="h-4 w-4" />}>
            New analysis
          </NavItem>
          <NavItem to="/compare" icon={<GitCompare className="h-4 w-4" />}>
            Compare
          </NavItem>
        </nav>
        <div className="mt-auto space-y-3 px-2 text-[11px] text-muted">
          {sys && (
            <div className="space-y-1 rounded-lg border border-edge bg-navy-850 p-3">
              <div className="flex justify-between">
                <span>Compute</span>
                <span className={sys.device === "cuda" ? "text-good" : "text-soft"}>{sys.gpu ?? sys.device.toUpperCase()}</span>
              </div>
              <div className="flex justify-between">
                <span>Pipeline</span>
                <span className="text-soft">v{sys.pipeline_version}</span>
              </div>
              <div className="flex justify-between">
                <span>Detector</span>
                <span className="text-soft">{sys.ultralytics ? "YOLO" : "MediaPipe"}</span>
              </div>
            </div>
          )}
          <button onClick={() => setSettings(true)} className="flex items-center gap-2 hover:text-accent">
            <KeyRound className="h-3.5 w-3.5" /> API key
          </button>
        </div>
      </aside>
      <div className="min-w-0 flex-1">
        <header className="flex items-center justify-between border-b border-edge px-4 py-3 md:hidden">
          <span className="font-bold text-white">BadmintonVision</span>
          <nav className="flex gap-3 text-sm text-muted">
            <NavLink to="/">Dashboard</NavLink>
            <NavLink to="/new">New</NavLink>
            <NavLink to="/compare">Compare</NavLink>
          </nav>
        </header>
        <main className="mx-auto max-w-[1500px] px-4 py-6 md:px-8 md:py-8">
          <Outlet />
        </main>
      </div>
      {settings && <SettingsDialog onClose={() => setSettings(false)} />}
    </div>
  );
}
