import { ScanSearch, UserCheck } from "lucide-react";
import { useState } from "react";
import { fmtTime } from "../lib/format";
import { api, mediaUrl } from "../services/api";
import type { Detections, Session } from "../types/api";
import { Button, Card, ErrorBox, Notice } from "./ui";

export function PlayerSelector({ session, onSelected }: { session: Session; onSelected: (s: Session) => void }) {
  const W = session.video.width;
  const H = session.video.height;
  const sel = session.player_selection;
  const [t, setT] = useState(sel?.t ?? Math.min(1, session.video.duration_s / 2));
  const [frameT, setFrameT] = useState(t);
  const [dets, setDets] = useState<Detections | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<unknown>(null);

  async function detect(at = frameT) {
    setBusy(true);
    setError(null);
    try {
      setDets(await api.detections(session.id, at));
    } catch (e) {
      setError(e);
      setDets(null);
    } finally {
      setBusy(false);
    }
  }

  async function choose(bbox: number[]) {
    if (!dets) return;
    const sx = W / dets.image_width;
    const sy = H / dets.image_height;
    try {
      const s = await api.selectPlayer(session.id, dets.t, [bbox[0] * sx, bbox[1] * sy, bbox[2] * sx, bbox[3] * sy]);
      onSelected(s);
    } catch (e) {
      setError(e);
    }
  }

  const boxes = dets?.people ?? [];
  const sx = dets ? W / dets.image_width : 1;
  const showSel = sel && Math.abs(sel.t - frameT) < 0.6;

  return (
    <Card
      title="Choose the player to analyse"
      subtitle="Detect people on a frame and click the player you want to follow. Tracking links them through occlusions."
      actions={
        <Button variant="primary" onClick={() => detect()} loading={busy}>
          <ScanSearch className="h-4 w-4" /> Detect players
        </Button>
      }
    >
      <div className="relative overflow-hidden rounded-xl border border-edge bg-black">
        <img src={mediaUrl(`/api/videos/${session.video.id}/frame`, { t: frameT, max_width: 1600 })} alt="Video frame" className="block w-full" draggable={false} />
        <svg viewBox={`0 0 ${W} ${H}`} preserveAspectRatio="none" className="absolute inset-0 h-full w-full">
          {boxes.map((p, i) => {
            const [x1, y1, x2, y2] = p.bbox.map((v) => v * sx);
            return (
              <g key={i} className="cursor-pointer" onClick={() => choose(p.bbox)}>
                <rect x={x1} y={y1} width={x2 - x1} height={y2 - y1} fill="rgba(34,211,238,0.08)" stroke="#22d3ee" strokeWidth={W / 500} className="transition hover:fill-[rgba(34,211,238,0.25)]" />
                <rect x={x1} y={y1 - H / 28} width={W / 9} height={H / 30} fill="#22d3ee" />
                <text x={x1 + W / 160} y={y1 - H / 80} fontSize={H / 42} fill="#0b1220" fontWeight={600}>
                  Select · {Math.round(p.score * 100)}%
                </text>
              </g>
            );
          })}
          {showSel && sel && (
            <rect x={sel.bbox[0]} y={sel.bbox[1]} width={sel.bbox[2] - sel.bbox[0]} height={sel.bbox[3] - sel.bbox[1]} fill="none" stroke="#a3e635" strokeWidth={W / 300} strokeDasharray={`${W / 120} ${W / 200}`} />
          )}
        </svg>
      </div>
      <div className="mt-4 flex items-center gap-3 text-xs text-muted">
        <span className="w-24">Frame {fmtTime(t)}</span>
        <input
          type="range"
          min={0}
          max={Math.max(0, session.video.duration_s - 0.1)}
          step={0.1}
          value={t}
          onChange={(e) => setT(Number(e.target.value))}
          onPointerUp={() => {
            setFrameT(t);
            setDets(null);
          }}
          className="flex-1"
        />
      </div>
      <div className="mt-3 space-y-2">
        {dets && boxes.length === 0 && <Notice>No people detected on this frame ({dets.detector}). Try another frame.</Notice>}
        {dets && boxes.length > 0 && <div className="text-xs text-muted">{boxes.length} people detected with {dets.detector}. Click a box to select.</div>}
        {sel ? (
          <div className="flex items-center gap-2 text-sm text-good">
            <UserCheck className="h-4 w-4" /> Player selected at {fmtTime(sel.t)} {showSel ? "(green dashed box)" : ""}
          </div>
        ) : (
          <div className="text-xs text-muted">No player selected yet. Without a selection the longest-tracked person on the near half is used.</div>
        )}
        {error ? <ErrorBox error={error} /> : null}
      </div>
    </Card>
  );
}
