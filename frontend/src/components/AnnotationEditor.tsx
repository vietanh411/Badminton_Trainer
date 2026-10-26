import { ChevronLeft, ChevronRight, Crosshair, Footprints, MousePointer2, Plus, Save, Square, Trash2, Wand2 } from "lucide-react";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  type AnnotationDoc,
  type AnnPerson,
  boxFromDrag,
  clientToImage,
  hitBox,
  nextPersonId,
  normaliseDoc,
  toSavedDoc,
} from "../lib/annotation";
import { fmtTime } from "../lib/format";
import { benchmarkApi, mediaUrl, poseApi } from "../services/api";
import { Badge, Button, Card, cx, ErrorBox, inputClass, Notice, Segmented, Spinner } from "./ui";

type Mode = "select" | "box" | "contact";

/**
 * Benchmark annotation editor: player boxes, identities and ground-contact points on sampled frames.
 * Frames come from the session's saved annotations or from the machine-generated template, and only
 * frames a human marks "reviewed" are saved as ground truth.
 */
export function AnnotationEditor({ sessionId, videoId, width, height, duration }: {
  sessionId: string;
  videoId: string;
  width: number;
  height: number;
  duration: number;
}) {
  const [doc, setDoc] = useState<AnnotationDoc | null>(null);
  const [error, setError] = useState<unknown>(null);
  const [idx, setIdx] = useState(0);
  const [sel, setSel] = useState(-1);
  const [mode, setMode] = useState<Mode>("select");
  const [drag, setDrag] = useState<{ a: [number, number]; b: [number, number] } | null>(null);
  const [saving, setSaving] = useState(false);
  const [saved, setSaved] = useState<string | null>(null);
  const [dirty, setDirty] = useState(false);
  const [newT, setNewT] = useState("");
  const svgRef = useRef<SVGSVGElement>(null);

  useEffect(() => {
    benchmarkApi
      .get(sessionId)
      .then((d) => setDoc(normaliseDoc(d)))
      .catch(setError);
  }, [sessionId]);

  const frame = doc?.frames[idx];
  const update = useCallback((fn: (d: AnnotationDoc) => AnnotationDoc) => {
    setDoc((d) => (d ? fn(d) : d));
    setDirty(true);
    setSaved(null);
  }, []);
  const updateFrame = useCallback(
    (fn: (people: AnnPerson[]) => AnnPerson[]) =>
      update((d) => ({ ...d, frames: d.frames.map((f, i) => (i === idx ? { ...f, people: fn(f.people) } : f)) })),
    [idx, update],
  );

  const toImage = (e: React.PointerEvent): [number, number] => {
    const rect = svgRef.current!.getBoundingClientRect();
    return clientToImage(e.clientX, e.clientY, rect, width, height);
  };

  const onDown = (e: React.PointerEvent) => {
    if (!frame) return;
    const p = toImage(e);
    if (mode === "box") {
      (e.target as Element).setPointerCapture?.(e.pointerId);
      setDrag({ a: p, b: p });
    } else if (mode === "contact") {
      if (sel < 0) return;
      updateFrame((people) => people.map((q, i) => (i === sel ? { ...q, ground_contact: p } : q)));
    } else {
      setSel(hitBox(frame.people, p));
    }
  };
  const onMove = (e: React.PointerEvent) => {
    if (drag) setDrag({ ...drag, b: toImage(e) });
  };
  const onUp = () => {
    if (!drag || !doc) return;
    const b = boxFromDrag(drag.a, drag.b, width, height);
    setDrag(null);
    if (!b) return;
    const id = nextPersonId(doc);
    updateFrame((people) => [...people, { id, bbox: b, ground_contact: null }]);
    setSel(frame ? frame.people.length : 0);
    setMode("select");
  };

  const go = (k: number) => {
    if (!doc) return;
    setIdx(Math.max(0, Math.min(doc.frames.length - 1, k)));
    setSel(-1);
  };

  const ids = useMemo(() => {
    const s = new Set<string>();
    doc?.frames.forEach((f) => f.people.forEach((p) => s.add(p.id)));
    return [...s].sort();
  }, [doc]);
  const reviewedCount = doc?.frames.filter((f) => f.reviewed).length ?? 0;

  if (error) return <ErrorBox error={error} />;
  if (!doc) return <Spinner label="Loading annotations…" />;

  const person = frame && sel >= 0 ? frame.people[sel] : undefined;

  return (
    <Card
      title="Benchmark annotation"
      subtitle="Draw tight player boxes (no racket), keep one id per real person, click the floor point between the feet. Only frames marked reviewed are used as ground truth."
      actions={
        <div className="flex items-center gap-2">
          <Badge tone={reviewedCount ? "good" : "neutral"}>{reviewedCount}/{doc.frames.length} reviewed</Badge>
          <Button
            variant="primary"
            loading={saving}
            disabled={!dirty}
            onClick={async () => {
              setSaving(true);
              setError(null);
              try {
                const r = await benchmarkApi.put(sessionId, toSavedDoc(doc));
                setSaved(`Saved ${r.frames} reviewed frame(s).`);
                setDirty(false);
              } catch (e) {
                setError(e);
              } finally {
                setSaving(false);
              }
            }}
          >
            <Save className="h-4 w-4" /> Save
          </Button>
        </div>
      }
    >
      {doc.source === "synthetic" && (
        <Notice tone="info">This demo session shows the simulator's ground truth. Annotation matters for real footage.</Notice>
      )}
      {saved && <Notice tone="info">{saved}</Notice>}
      <div className="mb-3 flex flex-wrap items-center gap-2">
        <Button
          onClick={async () => {
            try {
              const tpl = normaliseDoc(await poseApi.template(sessionId));
              update(() => tpl);
              setIdx(0);
              setSel(-1);
            } catch (e) {
              setError(e);
            }
          }}
          title="Replace with boxes from this session's own analysis (marked unreviewed)"
        >
          <Wand2 className="h-4 w-4" /> Load machine pre-annotation
        </Button>
        <input
          className={cx(inputClass, "w-24")}
          placeholder="time (s)"
          value={newT}
          onChange={(e) => setNewT(e.target.value)}
        />
        <Button
          onClick={() => {
            const t = Number(newT);
            if (!Number.isFinite(t) || t < 0 || t > duration) return;
            update((d) => ({ ...d, frames: [...d.frames, { t, people: [], reviewed: false }].sort((a, b) => a.t - b.t) }));
            setNewT("");
          }}
        >
          <Plus className="h-4 w-4" /> Add frame
        </Button>
        <div className="ml-auto">
          <Segmented<Mode>
            value={mode}
            onChange={setMode}
            options={[
              { value: "select", label: "Select" },
              { value: "box", label: "Draw box" },
              { value: "contact", label: "Ground contact" },
            ]}
          />
        </div>
      </div>

      {!frame ? (
        <Notice tone="info">No frames yet. Load the machine pre-annotation or add a frame by time.</Notice>
      ) : (
        <div className="grid gap-4 xl:grid-cols-[1fr_280px]">
          <div>
            <div className="relative overflow-hidden rounded-lg border border-edge bg-ink">
              <img
                src={mediaUrl(`/api/videos/${videoId}/frame`, { t: frame.t, max_width: Math.min(width, 1600) })}
                alt={`frame at ${fmtTime(frame.t)}`}
                className="block w-full select-none"
                draggable={false}
              />
              <svg
                ref={svgRef}
                viewBox={`0 0 ${width} ${height}`}
                className={cx("absolute inset-0 h-full w-full", mode === "select" ? "cursor-pointer" : "cursor-crosshair")}
                onPointerDown={onDown}
                onPointerMove={onMove}
                onPointerUp={onUp}
              >
                {frame.people.map((p, i) => {
                  const [x1, y1, x2, y2] = p.bbox;
                  const isTarget = p.id === doc.target_id;
                  const color = i === sel ? "#a3e635" : isTarget ? "#22d3ee" : "#c9d4e5";
                  return (
                    <g key={i}>
                      <rect x={x1} y={y1} width={x2 - x1} height={y2 - y1} fill="none" stroke={color} strokeWidth={Math.max(2, width / 500)} />
                      <text x={x1} y={y1 - 6} fill={color} fontSize={Math.max(14, width / 70)} fontWeight={600}>
                        {p.id}
                        {isTarget ? " (target)" : ""}
                      </text>
                      {p.ground_contact && (
                        <circle cx={p.ground_contact[0]} cy={p.ground_contact[1]} r={Math.max(5, width / 250)} fill="#fb923c" stroke="#070d18" />
                      )}
                    </g>
                  );
                })}
                {drag && (() => {
                  const [ax, ay] = drag.a;
                  const [bx, by] = drag.b;
                  return (
                    <rect x={Math.min(ax, bx)} y={Math.min(ay, by)} width={Math.abs(bx - ax)} height={Math.abs(by - ay)} fill="#22d3ee22" stroke="#22d3ee" strokeDasharray="6 4" />
                  );
                })()}
              </svg>
            </div>
            <div className="mt-2 flex items-center gap-2">
              <Button onClick={() => go(idx - 1)} disabled={idx === 0}>
                <ChevronLeft className="h-4 w-4" />
              </Button>
              <span className="tabular text-sm text-soft">
                Frame {idx + 1}/{doc.frames.length} · t = {fmtTime(frame.t)}
              </span>
              <Button onClick={() => go(idx + 1)} disabled={idx >= doc.frames.length - 1}>
                <ChevronRight className="h-4 w-4" />
              </Button>
              <label className="ml-auto flex items-center gap-2 text-sm text-soft">
                <input
                  type="checkbox"
                  checked={!!frame.reviewed}
                  onChange={(e) =>
                    update((d) => ({ ...d, frames: d.frames.map((f, i) => (i === idx ? { ...f, reviewed: e.target.checked } : f)) }))
                  }
                />
                Reviewed — boxes, ids and contacts are correct
              </label>
              <Button
                variant="danger"
                title="Remove this frame"
                onClick={() => {
                  update((d) => ({ ...d, frames: d.frames.filter((_, i) => i !== idx) }));
                  setIdx((k) => Math.max(0, k - 1));
                  setSel(-1);
                }}
              >
                <Trash2 className="h-4 w-4" />
              </Button>
            </div>
          </div>

          <div className="space-y-3 text-sm">
            <div className="rounded-lg border border-edge bg-navy-850 p-3">
              <p className="mb-2 flex items-center gap-2 font-semibold text-soft">
                <MousePointer2 className="h-4 w-4 text-accent" /> Selected person
              </p>
              {!person ? (
                <p className="text-xs text-muted">Click a box (Select mode) or draw one (Draw box mode).</p>
              ) : (
                <div className="space-y-2">
                  <label className="block text-xs text-muted">
                    Identity
                    <input
                      className={cx(inputClass, "mt-1")}
                      list="bv-ann-ids"
                      value={person.id}
                      onChange={(e) => updateFrame((people) => people.map((q, i) => (i === sel ? { ...q, id: e.target.value.trim() || q.id } : q)))}
                    />
                    <datalist id="bv-ann-ids">
                      {ids.map((v) => (
                        <option key={v} value={v} />
                      ))}
                    </datalist>
                  </label>
                  <div className="flex flex-wrap gap-2">
                    <Button onClick={() => update((d) => ({ ...d, target_id: person.id }))}>
                      <Crosshair className="h-4 w-4" /> Set as target
                    </Button>
                    <Button onClick={() => setMode("contact")}>
                      <Footprints className="h-4 w-4" /> Place contact
                    </Button>
                    <Button onClick={() => updateFrame((people) => people.map((q, i) => (i === sel ? { ...q, ground_contact: null } : q)))}>
                      Clear contact
                    </Button>
                    <label className="flex items-center gap-1 text-xs text-muted">
                      <input
                        type="checkbox"
                        checked={!!person.occluded}
                        onChange={(e) => updateFrame((people) => people.map((q, i) => (i === sel ? { ...q, occluded: e.target.checked } : q)))}
                      />
                      &gt; 50 % occluded
                    </label>
                    <Button
                      variant="danger"
                      onClick={() => {
                        updateFrame((people) => people.filter((_, i) => i !== sel));
                        setSel(-1);
                      }}
                    >
                      <Trash2 className="h-4 w-4" /> Delete box
                    </Button>
                  </div>
                  <p className="tabular text-xs text-muted">
                    box [{person.bbox.map((v) => v.toFixed(0)).join(", ")}] · contact {person.ground_contact ? person.ground_contact.map((v) => v.toFixed(0)).join(", ") : "none"}
                  </p>
                </div>
              )}
            </div>
            <div className="rounded-lg border border-edge bg-navy-850 p-3 text-xs text-muted">
              <p className="mb-1 flex items-center gap-2 font-semibold text-soft">
                <Square className="h-4 w-4 text-accent" /> Conventions
              </p>
              <ul className="list-disc space-y-1 pl-4">
                <li>Coordinates are stored in full-resolution video pixels ({width}×{height}).</li>
                <li>Contact: midpoint between both planted shoes; the planted shoe if one foot is up; none while airborne.</li>
                <li>Strokes are labelled in the Strokes tab; confirmed strokes are included when you load the pre-annotation.</li>
                <li>Unreviewed frames are kept but never scored, so the evaluation cannot grade the pipeline against itself.</li>
              </ul>
            </div>
          </div>
        </div>
      )}
    </Card>
  );
}
