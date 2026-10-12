import { useEffect, useRef, useState } from "react";
import type { PlayerContext } from "../shared/types";
import { ProgressPanel } from "./components/ProgressPanel";
import { Report } from "./components/Report";
import { UploadPanel } from "./components/UploadPanel";
import { runAnalysis, type AnalysisResult, type Progress } from "./lib/analyze";

type State =
  | { kind: "idle" }
  | { kind: "running"; progress: Progress }
  | { kind: "done"; result: AnalysisResult }
  | { kind: "error"; message: string };

export default function App() {
  const [state, setState] = useState<State>({ kind: "idle" });
  const [video, setVideo] = useState<{ file: File; url: string } | null>(null);
  const [context, setContext] = useState<PlayerContext>({
    hand: "auto",
    level: "intermediate",
    focus: "general",
    notes: "",
  });
  const abortRef = useRef<AbortController | null>(null);

  useEffect(() => () => abortRef.current?.abort(), []);

  const selectFile = (file: File | null) => {
    if (video) URL.revokeObjectURL(video.url);
    setVideo(file ? { file, url: URL.createObjectURL(file) } : null);
    setState({ kind: "idle" });
  };

  const start = async () => {
    if (!video) return;
    const controller = new AbortController();
    abortRef.current = controller;
    setState({ kind: "running", progress: { stage: "loading", fraction: null } });
    try {
      const result = await runAnalysis(
        video.url,
        context,
        (progress) => setState({ kind: "running", progress }),
        controller.signal,
      );
      setState({ kind: "done", result });
    } catch (err) {
      if (controller.signal.aborted) {
        setState({ kind: "idle" });
      } else {
        console.error(err);
        setState({ kind: "error", message: err instanceof Error ? err.message : String(err) });
      }
    }
  };

  const cancel = () => abortRef.current?.abort();
  const reset = () => selectFile(null);

  return (
    <div className="app">
      <header className="topbar">
        <div className="brand">
          <span className="brand-mark" aria-hidden>
            🏸
          </span>
          <span>Shuttle Coach</span>
        </div>
        {state.kind === "done" && (
          <button className="btn ghost" onClick={reset}>
            Analyze another video
          </button>
        )}
      </header>

      <main className="main">
        {state.kind === "done" && video ? (
          <Report result={state.result} videoUrl={video.url} />
        ) : state.kind === "running" ? (
          <ProgressPanel progress={state.progress} onCancel={cancel} />
        ) : (
          <>
            {state.kind === "error" && (
              <div className="alert" role="alert">
                <strong>Analysis failed.</strong> {state.message}
              </div>
            )}
            <UploadPanel
              file={video?.file ?? null}
              videoUrl={video?.url ?? null}
              onFile={selectFile}
              context={context}
              onContextChange={setContext}
              onStart={start}
            />
          </>
        )}
      </main>

      <footer className="footer">
        Pose tracking runs in your browser. Only a handful of still frames and measurements are sent for coaching.
      </footer>
    </div>
  );
}
