import { Film } from "lucide-react";
import { useState } from "react";
import { fmtTime } from "../lib/format";
import { annotatedVideoPath, downloadFile } from "../services/api";
import { Button, ErrorBox } from "./ui";

/** Renders a clip with overlays burned in on the server (synchronous, a few seconds) and downloads it. */
export function AnnotatedClipButton({ sessionId, range, label }: { sessionId: string; range: () => [number, number]; label?: string }) {
  const [busy, setBusy] = useState<[number, number] | null>(null);
  const [error, setError] = useState<unknown>(null);
  return (
    <div className="space-y-2">
      <Button
        loading={!!busy}
        onClick={async () => {
          const r = range();
          setBusy(r);
          setError(null);
          try {
            await downloadFile(annotatedVideoPath(sessionId, r[0], r[1]), `annotated_${sessionId.slice(0, 8)}.webm`);
          } catch (e) {
            setError(e);
          } finally {
            setBusy(null);
          }
        }}
      >
        <Film className="h-4 w-4" /> {busy ? `Rendering ${fmtTime(busy[0], 0)}–${fmtTime(busy[1], 0)}…` : label ?? "Download annotated clip"}
      </Button>
      {error ? <ErrorBox error={error} /> : null}
    </div>
  );
}
