/** Types mirroring the FastAPI responses (backend/app/api, app/services/serialize.py). */
import type { Mat3 } from "../lib/homography";

export type JobStatus =
  | "QUEUED"
  | "VALIDATING"
  | "CALIBRATING"
  | "DETECTING"
  | "TRACKING"
  | "ESTIMATING_POSE"
  | "ANALYZING"
  | "RENDERING"
  | "COMPLETED"
  | "FAILED"
  | "CANCELLED";

export const JOB_STAGES: JobStatus[] = [
  "QUEUED",
  "VALIDATING",
  "CALIBRATING",
  "DETECTING",
  "TRACKING",
  "ESTIMATING_POSE",
  "ANALYZING",
  "RENDERING",
  "COMPLETED",
];
export const TERMINAL: JobStatus[] = ["COMPLETED", "FAILED", "CANCELLED"];

export interface Video {
  id: string;
  original_filename: string;
  size_bytes: number;
  container: string;
  codec: string;
  width: number;
  height: number;
  fps: number;
  frame_count: number;
  duration_s: number;
  rotation: number;
  browser_playable: boolean;
  has_proxy: boolean;
  is_synthetic: boolean;
  created_at: string;
  preview_url: string;
  thumbnail_url: string;
}

export interface Calibration {
  id: string | null;
  method: string;
  frame_time_s: number;
  image_width: number;
  image_height: number;
  point_ids: string[];
  image_points: number[][];
  homography: Mat3;
  inliers: boolean[];
  errors_m: number[];
  errors_px: number[];
  mean_error_m: number;
  max_error_m: number;
  mean_error_px: number;
  condition_number: number;
  warnings: string[];
  quality: "good" | "fair" | "poor";
  saved: boolean;
}

export interface Job {
  id: string;
  session_id: string;
  status: JobStatus;
  stage: string | null;
  progress: number;
  message: string | null;
  error: string | null;
  created_at: string;
  started_at: string | null;
  finished_at: string | null;
  timings: Record<string, number> | null;
  settings?: Record<string, unknown>;
}

export interface SessionSummary {
  synthetic?: boolean;
  calibrated?: boolean;
  duration_s?: number;
  performance_mode?: string;
  detector?: string;
  target_coverage_pct?: number;
  strokes?: number | null;
  calibration_error_m?: number | null;
  throughput_fps?: number;
  distance_m?: number;
  distance_per_min_m?: number | null;
  mean_speed_mps?: number | null;
  max_speed_mps?: number | null;
  p95_speed_mps?: number | null;
  zones_visited_pct?: number;
  direction_changes?: number;
  observed_s?: number;
  recovery_time_median_s?: number | null;
  composite_score?: number | null;
}

export interface Session {
  id: string;
  name: string;
  notes: string | null;
  status: "draft" | "queued" | "processing" | "completed" | "failed" | "cancelled";
  data_source: "upload" | "synthetic";
  is_demo: boolean;
  created_at: string;
  updated_at: string;
  video_id: string;
  video: Video;
  settings: Partial<PipelineSettings>;
  player_selection: { t: number; bbox: number[]; auto?: boolean } | null;
  corrections: { t: number; track_id: number | null }[];
  calibration: Calibration | null;
  latest_job: Job | null;
  summary: SessionSummary;
}

export interface PipelineSettings {
  performance_mode: "fast" | "balanced" | "accurate";
  detector: "auto" | "yolo" | "mediapipe" | "truth";
  tracker: "bytetrack" | "botsort";
  analysis_fps: number | null;
  max_width: number | null;
  detect_strokes: boolean;
  heatmap: { cell_m: number; sigma_m: number; margin_m: number };
  [k: string]: unknown;
}

export interface Profile {
  analysis_fps: number;
  max_width: number;
  yolo_weights: string;
  yolo_imgsz: number;
  pose_model: string;
  description: string;
}

export interface SystemInfo {
  pipeline_version: string;
  python: string;
  opencv: string;
  mediapipe: string;
  device: string;
  gpu: string | null;
  torch: string | null;
  ultralytics: string | null;
  models: Record<string, boolean>;
  profiles: Record<string, Profile>;
  limits: {
    max_upload_mb: number;
    max_duration_s: number;
    max_concurrent_jobs: number;
    max_queued_jobs: number;
    allowed_extensions: string[];
  };
  auth_required: boolean;
  demo_enabled: boolean;
}

export interface CourtLine {
  id: string;
  label: string;
  start: [number, number];
  end: [number, number];
  kind: "boundary" | "sideline" | "service" | "center" | "net";
}

export interface CourtKeypoint {
  id: string;
  label: string;
  x: number;
  y: number;
}

export interface CourtGeometry {
  length_m: number;
  width_m: number;
  singles_width_m: number;
  net_y_m: number;
  lines: CourtLine[];
  keypoints: CourtKeypoint[];
}

export interface AutoCalibration {
  success: boolean;
  message: string;
  score: number;
  frame_time_s?: number;
  image_width?: number;
  image_height?: number;
  points?: { id: string; x: number; y: number }[];
  mean_error_m?: number;
  warnings?: string[];
}

export interface Detections {
  t: number;
  detector: string;
  image_width: number;
  image_height: number;
  people: { bbox: [number, number, number, number]; score: number }[];
}

/** Loosely typed metrics tree (see vision/analytics/movement.py). */
export interface Metrics {
  calibrated?: boolean;
  reason?: string;
  duration_s?: number;
  observed_s?: number;
  coverage_pct?: number;
  distance?: { total_m: number; interpolated_m: number; per_minute_m: number | null };
  speed?: {
    mean_mps: number | null;
    mean_while_moving_mps: number | null;
    p95_mps: number | null;
    max_filtered_mps: number | null;
    time_moving_s: number;
    time_stationary_s: number;
    moving_share: number | null;
    histogram: { bin_edges_mps: number[]; seconds: number[] };
  };
  zones?: {
    time_s: Record<string, number>;
    share: Record<string, number>;
    entries: Record<string, number>;
    outside_singles_s: number;
    visited: string[];
    visited_pct: number;
    frequented: string[];
    depth_share: Record<string, number>;
    lateral_share: Record<string, number>;
    convex_hull_area_m2: number | null;
  };
  recovery?: {
    base_position_half: { lateral_m: number; depth_from_net_m: number };
    mean_distance_from_base_m: number | null;
    time_far_from_base_s: number;
    far_share: number | null;
    excursions: number;
    recovery_time_median_s: number | null;
    label: string;
  };
  footwork?: {
    path_directness_mean: number | null;
    extra_distance_pct_mean: number | null;
    recovery_time_median_s: number | null;
    peak_acceleration_p95_mps2: number | null;
    peak_deceleration_p95_mps2: number | null;
    smoothness_ldlj_median: number | null;
    positioning_consistency_m: number | null;
  };
  composite?: {
    experimental: boolean;
    score: number | null;
    components: Record<string, number>;
    weights: Record<string, number>;
    formula: string;
    caveat: string;
  } | null;
  direction_changes?: number;
  direction_changes_per_minute?: number | null;
  heatmap_area_50_m2?: number;
  heatmap_area_90_m2?: number;
  data_quality?: {
    samples: number;
    with_position: number;
    measured: number;
    interpolated: number;
    rejected: Record<string, number>;
    segments: number;
  };
  pose?: { frames_with_pose: number; pose_coverage_pct: number; note?: string };
  strokes?: {
    candidates: number;
    handedness?: string | null;
    sampling_fps?: number;
    warning?: string | null;
    enabled?: boolean;
    status?: string;
    model?: Record<string, string>;
  };
}

export interface Quality {
  target_frames: number;
  target_coverage_pct: number;
  occluded_frames: number;
  linked_frames: number;
  identity_links: number;
  possible_id_switches: number;
  tracking_gaps: number;
  people_tracked: number;
  seed_track: number | null;
  selection: { auto: boolean; t: number };
  calibration: { source: string | null; mean_error_m?: number; warnings?: string[]; error?: string; score?: number };
}

export interface MetricsResponse {
  synthetic: boolean;
  metrics: Metrics;
  quality: Quality;
  analysis: {
    fps: number;
    width: number;
    height: number;
    scale: number;
    performance_mode: string;
    profile: Profile;
    detector: string;
    tracker: string;
    cache: { detections: boolean; pose: boolean };
  };
  video: Record<string, unknown>;
  timings_s: Record<string, number>;
  pipeline_version: string;
  settings: PipelineSettings;
  calibration: { homography: Mat3; image_size: [number, number] | null } | null;
  profiles: Record<string, Profile>;
}

export interface TrajectoryPoint {
  t: number;
  court_x: number | null;
  court_y: number | null;
  court_x_raw: number | null;
  court_y_raw: number | null;
  speed: number | null;
  interpolated: boolean | null;
  segment: number;
  position_source: string;
  position_confidence: number;
  reject_reason: string;
  zone: string | null;
  moving: boolean | null;
  distance_from_base_m: number | null;
}

export interface Trajectory {
  calibrated: boolean;
  step?: number;
  points: TrajectoryPoint[];
}

export interface HeatmapPayload {
  available: boolean;
  reason?: string;
  normalize?: "share" | "seconds" | "density";
  units?: string;
  cell_m?: number;
  sigma_m?: number;
  x0?: number;
  y0?: number;
  nx?: number;
  ny?: number;
  max?: number;
  total_s?: number;
  area_50_m2?: number;
  area_90_m2?: number;
  values?: number[][];
  observed_s?: number;
  synthetic?: boolean;
}

export interface AnnotationFrame {
  t: number;
  target: number | null;
  status: string;
  people: { id: number; b: [number, number, number, number]; s: number }[];
  pose: [number, number, number][] | null;
  foot: [number, number] | null;
  src: string;
  conf: number;
  court?: [number, number] | null;
  speed?: number | null;
  interp?: boolean;
}

export interface Annotations {
  start: number;
  end: number;
  analysis_width: number;
  analysis_height: number;
  scale: number;
  frames: AnnotationFrame[];
}

export interface Stroke {
  id: string;
  t: number;
  t_start: number | null;
  t_end: number | null;
  source: "detected" | "manual";
  review_status: "candidate" | "confirmed" | "rejected";
  confidence: number | null;
  shot_type: string | null;
  shot_type_confidence: number | null;
  stroke_side: string | null;
  stroke_side_confidence: number | null;
  height_class: string | null;
  classifier: string | null;
  signals: Record<string, number | null> | null;
  label_shot_type: string | null;
  label_stroke_side: string | null;
  experimental: boolean;
}

export interface TimelineEvent {
  type: string;
  t: number;
  t_end?: number;
  [k: string]: unknown;
}

export interface TrackSummary {
  track_id: number;
  first_t: number;
  last_t: number;
  n_frames: number;
  mean_score: number;
  is_target: boolean;
}

export interface CompareRow {
  key: string;
  label: string;
  unit: string;
  a: number | null;
  b: number | null;
  diff: number | null;
  rel_pct: number | null;
  direction: "more" | "less" | "neutral";
}

export interface CompareResponse {
  metrics: CompareRow[];
  quality: { key: string; label: string; a: unknown; b: unknown }[];
  warnings: string[];
  a: { id: string; name: string; synthetic: boolean };
  b: { id: string; name: string; synthetic: boolean };
}

export const SHOT_LABELS = ["smash", "clear", "drop", "drive", "net_shot", "lift", "other"] as const;
export type ShotLabel = (typeof SHOT_LABELS)[number];
