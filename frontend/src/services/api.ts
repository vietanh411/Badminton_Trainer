/** Typed client for the BadmintonVision FastAPI backend. */
import type {
  Annotations,
  AutoCalibration,
  Calibration,
  CompareResponse,
  CourtGeometry,
  Detections,
  HeatmapPayload,
  Job,
  MetricsResponse,
  PipelineSettings,
  Session,
  Stroke,
  SystemInfo,
  TimelineEvent,
  TrackSummary,
  Trajectory,
  Video,
} from "../types/api";

export const API_BASE: string = (import.meta.env.VITE_API_BASE as string | undefined) ?? "";
const KEY_STORAGE = "bv.apiKey";

export function getApiKey(): string | null {
  try {
    const stored = window.localStorage.getItem(KEY_STORAGE);
    if (stored) return stored;
  } catch {
    /* storage unavailable */
  }
  return (import.meta.env.VITE_API_KEY as string | undefined) || null;
}

export function setApiKey(key: string | null): void {
  try {
    if (key) window.localStorage.setItem(KEY_STORAGE, key);
    else window.localStorage.removeItem(KEY_STORAGE);
  } catch {
    /* storage unavailable */
  }
}

export class ApiError extends Error {
  constructor(
    public status: number,
    message: string,
  ) {
    super(message);
  }
}

function headers(extra?: HeadersInit): Headers {
  const h = new Headers(extra);
  const key = getApiKey();
  if (key) h.set("X-API-Key", key);
  return h;
}

/** URL for media elements (img/video/download links) — they cannot send headers, so the key goes in the query. */
export function mediaUrl(path: string, params?: Record<string, string | number | boolean | undefined | null>): string {
  const url = new URL(API_BASE + path, window.location.origin);
  for (const [k, v] of Object.entries(params ?? {})) if (v !== undefined && v !== null) url.searchParams.set(k, String(v));
  const key = getApiKey();
  if (key) url.searchParams.set("key", key);
  return API_BASE ? url.toString() : url.pathname + url.search;
}

async function detail(res: Response): Promise<string> {
  try {
    const body = (await res.json()) as { detail?: unknown };
    if (typeof body.detail === "string") return body.detail;
    if (Array.isArray(body.detail))
      return body.detail
        .map((d: { msg?: string; loc?: unknown[] }) => `${(d.loc ?? []).slice(1).join(".")}: ${d.msg ?? ""}`)
        .join("; ");
    return JSON.stringify(body.detail ?? body);
  } catch {
    return res.statusText || `HTTP ${res.status}`;
  }
}

export async function request<T>(path: string, init: RequestInit & { json?: unknown } = {}): Promise<T> {
  const { json, ...rest } = init;
  const h = headers(rest.headers);
  if (json !== undefined) h.set("Content-Type", "application/json");
  let res: Response;
  try {
    res = await fetch(API_BASE + path, { ...rest, headers: h, body: json !== undefined ? JSON.stringify(json) : rest.body });
  } catch {
    throw new ApiError(0, "Cannot reach the BadmintonVision API. Is the backend running on port 8000?");
  }
  if (!res.ok) throw new ApiError(res.status, await detail(res));
  if (res.status === 204) return undefined as T;
  return (await res.json()) as T;
}

function qs(params: Record<string, string | number | boolean | undefined | null>): string {
  const sp = new URLSearchParams();
  for (const [k, v] of Object.entries(params)) if (v !== undefined && v !== null && v !== "") sp.set(k, String(v));
  const s = sp.toString();
  return s ? `?${s}` : "";
}

/** Multipart upload with progress reporting (fetch has no upload progress events). */
export function uploadVideo(file: File, onProgress: (fraction: number) => void): Promise<{ video: Video; session: Session }> {
  return new Promise((resolve, reject) => {
    const xhr = new XMLHttpRequest();
    xhr.open("POST", API_BASE + "/api/videos");
    const key = getApiKey();
    if (key) xhr.setRequestHeader("X-API-Key", key);
    xhr.upload.onprogress = (e) => {
      if (e.lengthComputable) onProgress(e.loaded / e.total);
    };
    xhr.onload = () => {
      let body: unknown = null;
      try {
        body = JSON.parse(xhr.responseText);
      } catch {
        /* non-JSON */
      }
      if (xhr.status >= 200 && xhr.status < 300) resolve(body as { video: Video; session: Session });
      else {
        const d = (body as { detail?: unknown } | null)?.detail;
        reject(new ApiError(xhr.status, typeof d === "string" ? d : `Upload failed (HTTP ${xhr.status})`));
      }
    };
    xhr.onerror = () => reject(new ApiError(0, "Network error during upload"));
    const fd = new FormData();
    fd.append("file", file);
    xhr.send(fd);
  });
}

export const api = {
  system: () => request<SystemInfo>("/api/system"),
  court: () => request<CourtGeometry>("/api/court"),
  sessions: () => request<Session[]>("/api/sessions"),
  session: (id: string) => request<Session>(`/api/sessions/${id}`),
  patchSession: (id: string, body: { name?: string; notes?: string }) =>
    request<Session>(`/api/sessions/${id}`, { method: "PATCH", json: body }),
  deleteSession: (id: string) => request<void>(`/api/sessions/${id}`, { method: "DELETE" }),
  demo: (variant: 1 | 2) => request<{ session: Session; job: Job }>(`/api/demo${qs({ variant })}`, { method: "POST" }),

  calibrate: (
    id: string,
    body: { points: { id: string; x: number; y: number }[]; image_width: number; image_height: number; frame_time_s: number; dry_run: boolean },
  ) => request<Calibration>(`/api/sessions/${id}/calibration`, { method: "POST", json: body }),
  resetCalibration: (id: string) => request<void>(`/api/sessions/${id}/calibration`, { method: "DELETE" }),
  autoCalibrate: (id: string, t: number) =>
    request<AutoCalibration>(`/api/sessions/${id}/calibration/auto${qs({ t })}`, { method: "POST" }),
  detections: (id: string, t: number) => request<Detections>(`/api/sessions/${id}/detections${qs({ t })}`),
  selectPlayer: (id: string, t: number, bbox: number[]) =>
    request<Session>(`/api/sessions/${id}/player-selection`, { method: "POST", json: { t, bbox } }),
  tracks: (id: string) => request<TrackSummary[]>(`/api/sessions/${id}/tracks`),
  addCorrection: (id: string, t: number, track_id: number | null) =>
    request<{ corrections: Session["corrections"] }>(`/api/sessions/${id}/corrections`, { method: "POST", json: { t, track_id } }),
  clearCorrections: (id: string) => request<{ corrections: [] }>(`/api/sessions/${id}/corrections`, { method: "DELETE" }),

  analyze: (id: string, settings: Partial<PipelineSettings> & Record<string, unknown>) =>
    request<Job>(`/api/sessions/${id}/analyze`, { method: "POST", json: settings }),
  job: (id: string) => request<Job>(`/api/jobs/${id}`),
  cancelJob: (id: string) => request<Job>(`/api/jobs/${id}/cancel`, { method: "POST" }),

  metrics: (id: string) => request<MetricsResponse>(`/api/sessions/${id}/metrics`),
  trajectory: (id: string, p: { start?: number; end?: number; max_points?: number } = {}) =>
    request<Trajectory>(`/api/sessions/${id}/trajectory${qs(p)}`),
  heatmap: (
    id: string,
    p: { start?: number | null; end?: number | null; preset?: string; zone?: string; sigma_m?: number; cell_m?: number; normalize?: string },
  ) => request<HeatmapPayload>(`/api/sessions/${id}/heatmap${qs(p)}`),
  events: (id: string) => request<TimelineEvent[]>(`/api/sessions/${id}/events`),
  annotations: (id: string, start: number, end: number) =>
    request<Annotations>(`/api/sessions/${id}/annotations${qs({ start, end })}`),
  strokes: (id: string) => request<Stroke[]>(`/api/sessions/${id}/strokes`),
  addStroke: (id: string, body: { t: number; label_shot_type?: string | null; label_stroke_side?: string | null }) =>
    request<Stroke>(`/api/sessions/${id}/strokes`, { method: "POST", json: body }),
  patchStroke: (
    strokeId: string,
    body: { review_status?: string; t?: number; label_shot_type?: string | null; label_stroke_side?: string | null; clear_labels?: boolean },
  ) => request<Stroke>(`/api/strokes/${strokeId}`, { method: "PATCH", json: body }),
  deleteStroke: (strokeId: string) => request<void>(`/api/strokes/${strokeId}`, { method: "DELETE" }),
  compare: (a: string, b: string) => request<CompareResponse>(`/api/compare${qs({ a, b })}`),
  evaluation: (id: string) => request<Record<string, unknown>>(`/api/sessions/${id}/evaluation`),
};

export interface PoseSeries {
  t: number[];
  x_raw: (number | null)[];
  x_smooth: (number | null)[];
  y_raw: (number | null)[];
  y_smooth: (number | null)[];
  visibility: (number | null)[];
  units?: Record<string, string> | string;
  [k: string]: unknown;
}

export const poseApi = {
  series: (id: string, landmark: string, window_s = 0.3) =>
    request<PoseSeries>(`/api/sessions/${id}/pose-series${qs({ landmark, window_s })}`),
  template: (id: string) => request<Record<string, unknown>>(`/api/sessions/${id}/benchmark/template`),
};

/** Fetch a (possibly slow) file with the API key header and hand it to the browser as a download. */
export async function downloadFile(path: string, fallbackName: string): Promise<void> {
  let res: Response;
  try {
    res = await fetch(API_BASE + path, { headers: headers() });
  } catch {
    throw new ApiError(0, "Cannot reach the BadmintonVision API.");
  }
  if (!res.ok) throw new ApiError(res.status, await detail(res));
  const blob = await res.blob();
  const cd = res.headers.get("Content-Disposition") ?? "";
  const name = /filename="?([^";]+)"?/.exec(cd)?.[1] ?? fallbackName;
  saveBlob(blob, name);
}

export function saveBlob(blob: Blob, name: string): void {
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = name;
  document.body.appendChild(a);
  a.click();
  a.remove();
  window.setTimeout(() => URL.revokeObjectURL(url), 2000);
}

export function annotatedVideoPath(id: string, start: number, end: number): string {
  return `/api/sessions/${id}/annotated-video${qs({ start: start.toFixed(2), end: end.toFixed(2) })}`;
}

export function jobEventsUrl(jobId: string): string {
  return mediaUrl(`/api/jobs/${jobId}/events`);
}

export const benchmarkApi = {
  get: (id: string) => request<Record<string, unknown>>(`/api/sessions/${id}/benchmark`),
  put: (id: string, doc: unknown) =>
    request<{ saved: boolean; frames: number; strokes: number }>(`/api/sessions/${id}/benchmark`, { method: "PUT", json: doc }),
};
