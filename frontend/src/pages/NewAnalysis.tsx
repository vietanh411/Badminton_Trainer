import { CheckCircle2, FileVideo, UploadCloud } from "lucide-react";
import { useRef, useState } from "react";
import { useNavigate } from "react-router-dom";
import { Card, cx, ErrorBox } from "../components/ui";
import { useSystem } from "../hooks/useStatic";
import { fmtBytes } from "../lib/format";
import { uploadVideo } from "../services/api";

export function NewAnalysis() {
  const sys = useSystem();
  const navigate = useNavigate();
  const input = useRef<HTMLInputElement>(null);
  const [over, setOver] = useState(false);
  const [file, setFile] = useState<File | null>(null);
  const [progress, setProgress] = useState<number | null>(null);
  const [phase, setPhase] = useState<"idle" | "uploading" | "validating">("idle");
  const [error, setError] = useState<unknown>(null);
  const allowed = sys?.limits.allowed_extensions ?? [".mp4", ".mov", ".m4v", ".webm", ".mkv", ".avi"];
  const maxMb = sys?.limits.max_upload_mb ?? 1024;

  async function start(f: File) {
    setError(null);
    const ext = f.name.slice(f.name.lastIndexOf(".")).toLowerCase();
    if (!allowed.includes(ext)) {
      setError(new Error(`Unsupported file type "${ext}". Allowed: ${allowed.join(", ")}`));
      return;
    }
    if (f.size > maxMb * 1024 * 1024) {
      setError(new Error(`The file is ${fmtBytes(f.size)}; the server accepts up to ${maxMb} MB.`));
      return;
    }
    setFile(f);
    setPhase("uploading");
    setProgress(0);
    try {
      const r = await uploadVideo(f, (p) => {
        setProgress(p);
        if (p >= 1) setPhase("validating");
      });
      navigate(`/sessions/${r.session.id}/setup`);
    } catch (e) {
      setError(e);
      setPhase("idle");
      setProgress(null);
    }
  }

  return (
    <div className="mx-auto max-w-4xl space-y-6">
      <div>
        <h1 className="text-2xl font-bold tracking-tight text-white">New analysis</h1>
        <p className="mt-1 text-sm text-muted">Upload a singles match or practice clip. Next you'll calibrate the court and choose the player.</p>
      </div>
      <Card>
        <div
          onDragOver={(e) => {
            e.preventDefault();
            setOver(true);
          }}
          onDragLeave={() => setOver(false)}
          onDrop={(e) => {
            e.preventDefault();
            setOver(false);
            const f = e.dataTransfer.files[0];
            if (f && phase === "idle") void start(f);
          }}
          onClick={() => phase === "idle" && input.current?.click()}
          className={cx(
            "flex cursor-pointer flex-col items-center justify-center gap-4 rounded-2xl border-2 border-dashed px-6 py-16 text-center transition",
            over ? "border-accent bg-cyan-400/5" : "border-edge hover:border-cyan-400/50",
          )}
        >
          <input
            ref={input}
            type="file"
            accept={allowed.join(",") + ",video/*"}
            className="hidden"
            onChange={(e) => {
              const f = e.target.files?.[0];
              if (f) void start(f);
            }}
          />
          {phase === "idle" ? (
            <>
              <div className="grid h-16 w-16 place-items-center rounded-2xl bg-cyan-400/10 ring-1 ring-cyan-400/30">
                <UploadCloud className="h-8 w-8 text-accent" />
              </div>
              <div>
                <div className="text-lg font-semibold text-white">Drop a video here or click to browse</div>
                <div className="mt-1 text-sm text-muted">
                  {allowed.join(" ")} · up to {maxMb} MB · up to {Math.round((sys?.limits.max_duration_s ?? 1800) / 60)} min
                </div>
              </div>
            </>
          ) : (
            <div className="w-full max-w-md space-y-3">
              <div className="flex items-center gap-3 text-left">
                <FileVideo className="h-8 w-8 text-accent" />
                <div className="min-w-0 flex-1">
                  <div className="truncate font-medium text-white">{file?.name}</div>
                  <div className="text-xs text-muted">{fmtBytes(file?.size)}</div>
                </div>
                {phase === "validating" && <CheckCircle2 className="h-5 w-5 text-good" />}
              </div>
              <div className="h-2 overflow-hidden rounded-full bg-navy-700">
                <div className="h-full rounded-full bg-accent transition-all" style={{ width: `${(progress ?? 0) * 100}%` }} />
              </div>
              <div className="text-xs text-muted">
                {phase === "uploading" ? `Uploading… ${Math.round((progress ?? 0) * 100)}%` : "Checking the file, reading metadata and creating a thumbnail…"}
              </div>
            </div>
          )}
        </div>
        {error ? <div className="mt-4"><ErrorBox error={error} /></div> : null}
      </Card>
      <Card title="Filming for reliable results">
        <ul className="grid gap-3 text-sm text-muted sm:grid-cols-2">
          <li>• Stationary camera (tripod) behind the baseline or at a high corner.</li>
          <li>• The selected player's full body — including the feet — visible most of the time.</li>
          <li>• Court lines visible enough to click at least 4 landmarks (6+ is better).</li>
          <li>• 720p or 1080p, 24–60 fps, indoor lighting. Singles is the supported format.</li>
        </ul>
        <p className="mt-3 text-xs text-muted">
          Doubles, moving cameras and heavily occluded players are handled gracefully but results are less reliable; the data-quality
          panel reports tracking coverage and possible identity switches.
        </p>
      </Card>
    </div>
  );
}
