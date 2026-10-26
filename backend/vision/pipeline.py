"""End-to-end analysis pipeline (framework-free).

VALIDATING → CALIBRATING → DETECTING (+ online tracking) → TRACKING (target identity) →
ESTIMATING_POSE → ANALYZING (positions, trajectory, metrics, strokes, heatmap) → RENDERING.

Expensive stages are cached on disk under a key derived from the video's content hash and the
settings that influence them, so re-running with different analytics settings, a corrected
calibration or a different target correction does not re-run detection.
"""

from __future__ import annotations

import hashlib
import json
import threading
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field, fields, is_dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from vision.analytics.heatmap import HeatmapConfig, compute_heatmap
from vision.analytics.movement import MovementConfig, compute_movement_metrics
from vision.calibration.auto import auto_calibrate
from vision.calibration.homography import Calibration, project
from vision.detection.base import Detection, filter_detections
from vision.geometry.court import COURT_LENGTH, COURT_WIDTH, NET_Y
from vision.pose.landmarks import NUM_LANDMARKS, body_metrics
from vision.strokes.detection import StrokeConfig, detect_strokes
from vision.tracking.appearance import HIST_DIM, appearance_descriptor
from vision.tracking.bytetrack import ByteTrackConfig, ByteTracker
from vision.tracking.target import TargetConfig, TargetSelectionError, resolve_target
from vision.trajectory.filtering import TrajectoryConfig, process_trajectory, sample_weights
from vision.trajectory.position import foot_contact_points, to_court
from vision.video.reader import iter_frames, probe_video, read_frame_at, resize_to_width

PIPELINE_VERSION = "1.0.0"
STAGES = ["VALIDATING", "CALIBRATING", "DETECTING", "TRACKING", "ESTIMATING_POSE", "ANALYZING", "RENDERING"]

PROFILES: dict[str, dict] = {
    "fast": {
        "analysis_fps": 10.0,
        "max_width": 640,
        "yolo_weights": "yolo11n.pt",
        "yolo_imgsz": 640,
        "pose_model": "pose_landmarker_lite.task",
        "description": "10 fps, 640 px, nano detector, lite pose",
    },
    "balanced": {
        "analysis_fps": 15.0,
        "max_width": 960,
        "yolo_weights": "yolo11n.pt",
        "yolo_imgsz": 640,
        "pose_model": "pose_landmarker_full.task",
        "description": "15 fps, 960 px, nano detector, full pose",
    },
    "accurate": {
        "analysis_fps": 30.0,
        "max_width": 1280,
        "yolo_weights": "yolo11s.pt",
        "yolo_imgsz": 960,
        "pose_model": "pose_landmarker_heavy.task",
        "description": "30 fps, 1280 px, small detector, heavy pose",
    },
}
MEDIAPIPE_DETECTOR = "efficientdet_lite0.tflite"


class Cancelled(Exception):
    pass


class PipelineError(RuntimeError):
    pass


@dataclass
class PipelineSettings:
    performance_mode: str = "balanced"
    detector: str = "auto"  # auto | yolo | mediapipe | truth (synthetic only)
    tracker: str = "bytetrack"  # bytetrack (built-in) | botsort (Ultralytics, needs yolo)
    analysis_fps: float | None = None
    max_width: int | None = None
    pose_running_mode: str = "image"
    detect_strokes: bool = True
    trajectory: TrajectoryConfig = field(default_factory=TrajectoryConfig)
    movement: MovementConfig = field(default_factory=MovementConfig)
    heatmap: HeatmapConfig = field(default_factory=HeatmapConfig)
    strokes: StrokeConfig = field(default_factory=StrokeConfig)
    target: TargetConfig = field(default_factory=TargetConfig)
    bytetrack: ByteTrackConfig = field(default_factory=ByteTrackConfig)

    def __post_init__(self) -> None:
        if self.performance_mode not in PROFILES:
            raise ValueError(f"unknown performance mode {self.performance_mode}")

    @property
    def profile(self) -> dict:
        p = dict(PROFILES[self.performance_mode])
        if self.analysis_fps:
            p["analysis_fps"] = float(self.analysis_fps)
        if self.max_width:
            p["max_width"] = int(self.max_width)
        return p

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict | None) -> PipelineSettings:
        d = dict(d or {})
        kwargs = {}
        for f in fields(cls):
            if f.name not in d:
                continue
            v = d[f.name]
            default = f.default_factory() if callable(f.default_factory) else None  # type: ignore[misc]
            if is_dataclass(default) and isinstance(v, dict):
                known = {k: val for k, val in v.items() if k in {ff.name for ff in fields(type(default))}}
                for k, val in list(known.items()):
                    if isinstance(getattr(default, k), tuple) and isinstance(val, list):
                        known[k] = tuple(val)
                kwargs[f.name] = type(default)(**{**asdict(default), **known})
            else:
                kwargs[f.name] = v
        return cls(**kwargs)


@dataclass
class PipelineInputs:
    video_path: Path
    video_hash: str
    out_dir: Path
    cache_dir: Path
    models_dir: Path
    settings: PipelineSettings
    calibration: dict | None = None  # Calibration.to_dict(), in original-video pixels
    selection: dict | None = None  # {"t": s, "bbox": [x1, y1, x2, y2]} in original-video pixels
    corrections: list[dict] = field(default_factory=list)
    synthetic_truth: Path | None = None
    stroke_model_dir: Path | None = None


ProgressFn = Callable[[str, float, str], None]


def _key(*parts) -> str:
    h = hashlib.sha256(json.dumps(parts, sort_keys=True, default=str).encode()).hexdigest()
    return h[:20]


def _check(cancel: threading.Event | None) -> None:
    if cancel is not None and cancel.is_set():
        raise Cancelled()


# ----------------------------------------------------------------------------- model setup


def resolve_detector_name(settings: PipelineSettings, models_dir: Path, synthetic: bool) -> str:
    if settings.detector == "truth":
        if not synthetic:
            raise PipelineError("the synthetic ground-truth detector is only allowed for synthetic demo videos")
        return "truth"
    if settings.detector in ("yolo", "mediapipe"):
        return settings.detector
    if synthetic:
        return "truth"
    try:
        import ultralytics  # noqa: F401

        return "yolo"
    except ImportError:
        return "mediapipe"


def build_detector(name: str, settings: PipelineSettings, models_dir: Path, truth: Path | None):
    from vision.detection.backends import MediaPipeDetector, TruthDetector, YoloDetector

    prof = settings.profile
    if name == "truth":
        return TruthDetector(truth, seed=1)
    if name == "yolo":
        w = models_dir / prof["yolo_weights"]
        return YoloDetector(w if w.exists() else prof["yolo_weights"], imgsz=prof["yolo_imgsz"])
    if name == "mediapipe":
        path = models_dir / MEDIAPIPE_DETECTOR
        if not path.exists():
            raise PipelineError(f"missing model {path.name}; run scripts/download_models.py")
        return MediaPipeDetector(path)
    raise PipelineError(f"unknown detector {name}")


def build_pose(settings: PipelineSettings, models_dir: Path, truth: Path | None, synthetic_pose: bool):
    from vision.pose.estimator import MediaPipePoseEstimator, TruthPoseEstimator

    if synthetic_pose:
        return TruthPoseEstimator(truth, seed=2)
    path = models_dir / settings.profile["pose_model"]
    if not path.exists():
        raise PipelineError(f"missing model {path.name}; run scripts/download_models.py")
    return MediaPipePoseEstimator(path, running_mode=settings.pose_running_mode)


# ------------------------------------------------------------------------------- stages


def stage_detect_track(inp: PipelineInputs, detector_name: str, progress: ProgressFn, cancel) -> dict:
    s = inp.settings
    prof = s.profile
    key = _key(
        "det",
        PIPELINE_VERSION,
        inp.video_hash,
        detector_name,
        s.tracker,
        prof["analysis_fps"],
        prof["max_width"],
        prof.get("yolo_weights"),
        prof.get("yolo_imgsz"),
        asdict(s.bytetrack),
    )
    cdir = inp.cache_dir / f"det_{key}"
    if (cdir / "tracks.parquet").exists():
        progress("DETECTING", 1.0, "re-using cached detections and tracks")
        return {
            "key": key,
            "samples": pd.read_parquet(cdir / "samples.parquet"),
            "tracks": pd.read_parquet(cdir / "tracks.parquet"),
            "detections": pd.read_parquet(cdir / "detections.parquet"),
            "appearance": np.load(cdir / "appearance.npy"),
            "cached": True,
        }

    meta = probe_video(inp.video_path)
    samples, det_rows, track_rows, app_rows = [], [], [], []
    started = time.time()
    if s.tracker == "botsort":
        if detector_name != "yolo":
            raise PipelineError("BoT-SORT requires the YOLO detector")
        from vision.tracking.ultralytics_tracker import UltralyticsTracker

        w = inp.models_dir / prof["yolo_weights"]
        combined = UltralyticsTracker(w if w.exists() else prof["yolo_weights"], "botsort", imgsz=prof["yolo_imgsz"])
        detector, tracker = None, None
    else:
        combined = None
        detector = build_detector(detector_name, s, inp.models_dir, inp.synthetic_truth)
        tracker = ByteTracker(fps=prof["analysis_fps"], config=s.bytetrack)
    try:
        for frame in iter_frames(
            inp.video_path, target_fps=prof["analysis_fps"], max_width=prof["max_width"], cancel=cancel
        ):
            _check(cancel)
            samples.append((frame.sample_index, frame.index, frame.timestamp_s))
            if combined is not None:
                dets, outs = combined.step(frame)
                dets = filter_detections(dets, frame_height=frame.image.shape[0])
            else:
                dets = filter_detections(detector.detect(frame), frame_height=frame.image.shape[0])
                outs = tracker.update(dets)
            for d in dets:
                det_rows.append((frame.sample_index, *d.bbox.tolist(), d.score))
            for o in outs:
                track_rows.append(
                    (frame.sample_index, frame.index, frame.timestamp_s, o.track_id, *o.bbox.tolist(), o.score)
                )
                app_rows.append(appearance_descriptor(frame.image, o.bbox))
            if frame.sample_index % 10 == 0:
                frac = min(1.0, frame.timestamp_s / max(meta.duration_s, 1e-6))
                rate = (frame.sample_index + 1) / max(time.time() - started, 1e-6)
                progress("DETECTING", frac, f"{frame.timestamp_s:.1f}s / {meta.duration_s:.1f}s · {rate:.1f} frames/s")
    finally:
        for obj in (detector, combined):
            if obj is not None:
                obj.close()
    if not samples:
        raise PipelineError("no frames could be decoded")
    samples_df = pd.DataFrame(samples, columns=["sample_index", "frame_index", "t"])
    tracks_df = pd.DataFrame(
        track_rows, columns=["sample_index", "frame_index", "t", "track_id", "x1", "y1", "x2", "y2", "score"]
    )
    dets_df = pd.DataFrame(det_rows, columns=["sample_index", "x1", "y1", "x2", "y2", "score"])
    app = np.array(app_rows, np.float32).reshape(-1, HIST_DIM)
    cdir.mkdir(parents=True, exist_ok=True)
    samples_df.to_parquet(cdir / "samples.parquet")
    tracks_df.to_parquet(cdir / "tracks.parquet")
    dets_df.to_parquet(cdir / "detections.parquet")
    np.save(cdir / "appearance.npy", app)
    (cdir / "info.json").write_text(
        json.dumps(
            {
                "detector": detector_name,
                "tracker": s.tracker,
                "profile": prof,
                "seconds": time.time() - started,
                "samples": len(samples_df),
            }
        )
    )
    return {
        "key": key,
        "samples": samples_df,
        "tracks": tracks_df,
        "detections": dets_df,
        "appearance": app,
        "cached": False,
    }


def _court_filter_tracks(tracks: pd.DataFrame, H: np.ndarray | None, margin: float = 2.0) -> pd.DataFrame:
    """Drop whole tracks whose median foot point lies off the court (umpires, coaches, spectators)."""
    if H is None or tracks.empty:
        return tracks
    foot = np.column_stack([(tracks["x1"] + tracks["x2"]) / 2, tracks["y2"]])
    xy = project(H, foot)
    df = tracks.assign(cx=xy[:, 0], cy=xy[:, 1])
    med = df.groupby("track_id")[["cx", "cy"]].median()
    ok = med[
        (med.cx > -margin)
        & (med.cx < COURT_WIDTH + margin)
        & (med.cy > -margin - 1)
        & (med.cy < COURT_LENGTH + margin + 1)
    ].index
    return tracks[tracks["track_id"].isin(ok)]


def auto_select_target(tracks: pd.DataFrame, H: np.ndarray | None) -> dict:
    """Fallback when the user did not pick a player: the longest-tracked person on the near half."""
    g = tracks.groupby("track_id")
    stats = g.agg(n=("t", "size"), h=("y2", "median"), t=("t", "median"))
    if H is not None:
        foot = np.column_stack([(tracks["x1"] + tracks["x2"]) / 2, tracks["y2"]])
        y = project(H, foot)[:, 1]
        near = pd.Series(y < NET_Y, index=tracks.index).groupby(tracks["track_id"]).mean()
        stats["near"] = near
        cand = stats[stats["near"] > 0.5]
        stats = cand if not cand.empty else stats
    best = stats.sort_values("n", ascending=False).index[0]
    row = tracks[tracks["track_id"] == best].iloc[len(tracks[tracks["track_id"] == best]) // 2]
    return {"t": float(row["t"]), "bbox": [row["x1"], row["y1"], row["x2"], row["y2"]], "auto": True}


def stage_pose(
    inp: PipelineInputs, det_key: str, target: pd.DataFrame, synthetic_pose: bool, progress: ProgressFn, cancel
):
    s = inp.settings
    prof = s.profile
    assignment = target[["sample_index", "x1", "y1", "x2", "y2"]].dropna().round(1).to_numpy().tolist()
    key = _key(
        "pose",
        PIPELINE_VERSION,
        det_key,
        prof["pose_model"],
        s.pose_running_mode,
        synthetic_pose,
        hashlib.sha256(json.dumps(assignment).encode()).hexdigest(),
    )
    cpath = inp.cache_dir / f"pose_{key}.npz"
    n = len(target)
    if cpath.exists():
        progress("ESTIMATING_POSE", 1.0, "re-using cached pose landmarks")
        z = np.load(cpath)
        return {k: z[k] for k in z.files} | {"key": key, "cached": True}
    xy = np.full((n, NUM_LANDMARKS, 2), np.nan, np.float32)
    zz = np.full((n, NUM_LANDMARKS), np.nan, np.float32)
    vis = np.zeros((n, NUM_LANDMARKS), np.float32)
    pres = np.zeros((n, NUM_LANDMARKS), np.float32)
    boxes = {int(r.sample_index): np.array([r.x1, r.y1, r.x2, r.y2]) for r in target.dropna(subset=["x1"]).itertuples()}
    pos = {int(si): i for i, si in enumerate(target["sample_index"])}
    est = build_pose(s, inp.models_dir, inp.synthetic_truth, synthetic_pose)
    done = 0
    try:
        for frame in iter_frames(
            inp.video_path, target_fps=prof["analysis_fps"], max_width=prof["max_width"], cancel=cancel
        ):
            _check(cancel)
            b = boxes.get(frame.sample_index)
            if b is None:
                continue
            res = est.estimate(frame, b)
            done += 1
            if res is not None:
                i = pos[frame.sample_index]
                xy[i], zz[i], vis[i], pres[i] = res.xy, res.z, res.visibility, res.presence
            if done % 10 == 0:
                progress("ESTIMATING_POSE", done / max(1, len(boxes)), f"{done}/{len(boxes)} target frames")
    finally:
        est.close()
    np.savez_compressed(cpath, xy=xy, z=zz, visibility=vis, presence=pres)
    return {"xy": xy, "z": zz, "visibility": vis, "presence": pres, "key": key, "cached": False}


def analyse(
    target: pd.DataFrame,
    pose: dict,
    H: np.ndarray | None,
    settings: PipelineSettings,
    stroke_model_dir: Path | None = None,
) -> dict:
    """Positions → trajectory → movement metrics → strokes → heatmap. Pure computation."""
    t = target["t"].to_numpy(float)
    bbox = target[["x1", "y1", "x2", "y2"]].to_numpy(float)
    xy = pose["xy"].astype(float)
    vis = pose["visibility"].astype(float)
    contact = foot_contact_points(xy, vis, bbox, target["score"].to_numpy(float))
    out: dict = {"calibrated": H is not None}
    positions = pd.concat(
        [
            target[
                ["sample_index", "frame_index", "t", "track_id", "status", "x1", "y1", "x2", "y2", "max_overlap_iou"]
            ].reset_index(drop=True),
            contact,
        ],
        axis=1,
    )
    depth_from_net = None
    if H is not None:
        contact = to_court(contact, H)
        traj = process_trajectory(
            t,
            contact["court_x_raw"].to_numpy(),
            contact["court_y_raw"].to_numpy(),
            contact["position_confidence"].to_numpy(),
            settings.trajectory,
        )
        metrics, movement_events, derived = compute_movement_metrics(traj, settings.movement)
        positions = pd.concat(
            [positions, traj.drop(columns=["t", "position_confidence"]), derived.drop(columns=["t"])], axis=1
        )
        depth_from_net = derived["depth_m"].to_numpy()
        w = sample_weights(
            t, np.isfinite(traj["court_x"]).to_numpy() & (traj["segment"] >= 0).to_numpy(), traj["segment"].to_numpy()
        )
        hm = compute_heatmap(traj["court_x"].to_numpy(), traj["court_y"].to_numpy(), w, settings.heatmap)
        metrics["heatmap_area_50_m2"] = hm.area_for_share(0.5)
        metrics["heatmap_area_90_m2"] = hm.area_for_share(0.9)
        out["heatmap"] = hm
    else:
        metrics = {
            "calibrated": False,
            "reason": "No valid court calibration: court positions, distances, speeds, zones and heatmaps "
            "need a homography. Calibrate the court and re-run the analysis.",
        }
        movement_events = []
    # Pose summary (monocular proxies).
    pose_ok = np.all(np.isfinite(xy), axis=(1, 2))
    bm = body_metrics(xy, vis)
    pose_summary = {
        "frames_with_pose": int(pose_ok.sum()),
        "pose_coverage_pct": 100.0 * pose_ok.sum() / max(1, target["track_id"].notna().sum()),
    }
    for k in (
        "left_knee_flexion",
        "right_knee_flexion",
        "left_elbow_angle",
        "right_elbow_angle",
        "stance_width",
        "torso_lean_deg",
    ):
        v = bm[k][pose_ok]
        v = v[np.isfinite(v)]
        pose_summary[k] = {"median": float(np.median(v)), "p90": float(np.percentile(v, 90))} if v.size else None
    for k in (
        "left_knee_flexion",
        "right_knee_flexion",
        "left_elbow_angle",
        "right_elbow_angle",
        "stance_width",
        "torso_lean_deg",
        "left_arm_elevation",
        "right_arm_elevation",
        "torso_orientation",
    ):
        positions[f"pose_{k}"] = bm[k]
    metrics["pose"] = pose_summary
    metrics["pose"]["note"] = "2-D monocular proxies; not calibrated joint angles"

    stroke_events, stroke_diag = [], {"enabled": False}
    if settings.detect_strokes and pose_ok.sum() >= 10:
        stroke_events, stroke_diag = detect_strokes(
            t, np.where(pose_ok[:, None, None], xy, np.nan), vis, settings.strokes, depth_from_net
        )
        stroke_diag["enabled"] = True
        if stroke_model_dir is not None and stroke_events:
            stroke_diag["model"] = apply_stroke_model(
                stroke_model_dir, stroke_events, t, xy, vis, stroke_diag["handedness"]
            )
    metrics["strokes"] = {
        "candidates": len(stroke_events),
        **stroke_diag,
        "status": "event detection supported; classification experimental",
    }
    out.update(
        {
            "positions": positions,
            "metrics": metrics,
            "events": movement_events + [{"type": "stroke", **e} for e in stroke_events],
        }
    )
    return out


def apply_stroke_model(model_dir: Path, events: list[dict], t, xy, vis, hand) -> dict:
    from vision.strokes.classifier import TrainedStrokeModel, event_features

    used = {}
    for target_name in ("shot_type", "stroke_side"):
        d = model_dir / target_name
        if not (d / "model.joblib").exists():
            continue
        model = TrainedStrokeModel.load(d)
        X = np.vstack([event_features(t, xy, vis, e["t"], hand) for e in events])
        for e, pred in zip(events, model.predict(X), strict=True):
            e[target_name] = pred["label"]
            e[f"{target_name}_confidence"] = round(pred["confidence"], 3)
            e["classifier"] = model.version
        used[target_name] = model.version
    return used


# ------------------------------------------------------------------------------ runner


def load_calibration(
    inp: PipelineInputs, analysis_size: tuple[int, int], progress: ProgressFn
) -> tuple[Calibration | None, dict]:
    info: dict = {}
    if inp.calibration:
        cal = Calibration.from_dict(inp.calibration)
        info = {"source": cal.method, "mean_error_m": cal.mean_error_m, "warnings": cal.warnings}
    else:
        progress("CALIBRATING", 0.3, "no manual calibration; trying automatic court detection")
        meta = probe_video(inp.video_path)
        img, _ = read_frame_at(inp.video_path, min(1.0, meta.duration_s / 2))
        res = auto_calibrate(img)
        if not res.success:
            return None, {"source": None, "error": f"automatic calibration failed: {res.message}"}
        cal = res.calibration
        info = {
            "source": "auto (unverified)",
            "score": res.score,
            "mean_error_m": cal.mean_error_m,
            "warnings": cal.warnings,
            "calibration": cal.to_dict(),
        }
    return cal, info


def run_pipeline(
    inp: PipelineInputs, progress: ProgressFn | None = None, cancel: threading.Event | None = None
) -> dict:
    progress = progress or (lambda stage, frac, msg: None)
    t0 = time.time()
    timings: dict[str, float] = {}
    inp.out_dir.mkdir(parents=True, exist_ok=True)
    inp.cache_dir.mkdir(parents=True, exist_ok=True)
    s = inp.settings
    prof = s.profile

    progress("VALIDATING", 0.0, "reading video metadata")
    meta = probe_video(inp.video_path)
    first, _ = read_frame_at(inp.video_path, 0.0)
    analysis_img, scale = resize_to_width(first, prof["max_width"])
    ah, aw = analysis_img.shape[:2]
    timings["validating"] = time.time() - t0
    _check(cancel)

    t1 = time.time()
    progress("CALIBRATING", 0.0, "loading court calibration")
    cal, cal_info = load_calibration(inp, (aw, ah), progress)
    H = cal.scaled_to(aw, ah) if cal is not None else None
    if cal is not None and cal.image_size is None:
        H = cal.H @ np.diag([1 / scale, 1 / scale, 1.0])
    timings["calibrating"] = time.time() - t1
    _check(cancel)

    t1 = time.time()
    synthetic = inp.synthetic_truth is not None
    det_name = resolve_detector_name(s, inp.models_dir, synthetic)
    progress("DETECTING", 0.0, f"detecting people with {det_name}")
    det = stage_detect_track(inp, det_name, progress, cancel)
    timings["detecting"] = time.time() - t1

    t1 = time.time()
    progress("TRACKING", 0.0, "resolving the selected player's identity")
    tracks = _court_filter_tracks(det["tracks"], H)
    if tracks.empty and not det["tracks"].empty and cal is not None and not inp.calibration:
        # An unverified automatic calibration that puts every person off the court is wrong.
        cal_info = {
            "source": None,
            "error": "automatic calibration rejected: it placed every tracked person off the court",
        }
        cal, H = None, None
        tracks = det["tracks"]
    app = det["appearance"][tracks.index.to_numpy()] if len(tracks) else det["appearance"][:0]
    tracks = tracks.reset_index(drop=True)
    if tracks.empty:
        raise PipelineError("no people were tracked on the court; check the video and calibration")
    if inp.selection:
        sel = {"t": float(inp.selection["t"]), "bbox": (np.asarray(inp.selection["bbox"], float) * scale).tolist()}
    else:
        sel = auto_select_target(tracks, H)
    try:
        tres = resolve_target(tracks, app, det["samples"], sel, H=H, corrections=inp.corrections, config=s.target)
    except TargetSelectionError as exc:
        raise PipelineError(str(exc)) from exc
    target = tres.frames
    timings["tracking"] = time.time() - t1
    _check(cancel)

    t1 = time.time()
    progress("ESTIMATING_POSE", 0.0, "estimating pose of the selected player")
    synthetic_pose = synthetic and det_name == "truth"
    pose = stage_pose(inp, det["key"], target, synthetic_pose, progress, cancel)
    timings["pose"] = time.time() - t1
    _check(cancel)

    t1 = time.time()
    progress("ANALYZING", 0.0, "computing trajectory and movement metrics")
    res = analyse(target, pose, H, s, inp.stroke_model_dir)
    timings["analyzing"] = time.time() - t1
    _check(cancel)

    t1 = time.time()
    progress("RENDERING", 0.0, "writing results")
    out = inp.out_dir
    res["positions"].to_parquet(out / "positions.parquet")
    det["tracks"].to_parquet(out / "tracks.parquet")
    det["detections"].to_parquet(out / "detections.parquet")
    det["samples"].to_parquet(out / "samples.parquet")
    np.savez_compressed(
        out / "pose.npz", xy=pose["xy"], z=pose["z"], visibility=pose["visibility"], presence=pose["presence"]
    )
    events = sorted(tres.events + res["events"], key=lambda e: e["t"])
    (out / "events.json").write_text(json.dumps(clean_json(events), default=_json_default))
    if "heatmap" in res:
        from vision.visualization.court_plot import heatmap_figure, save_figure

        hm = res["heatmap"]
        traj = res["positions"][["court_x", "court_y"]].to_numpy(float)
        save_figure(
            heatmap_figure(hm, title="Court occupancy (share of observed time)", trajectory=traj), out / "heatmap.png"
        )
        save_figure(
            heatmap_figure(hm, title="Court occupancy (share of observed time)", trajectory=traj), out / "heatmap.svg"
        )
    timings["rendering"] = time.time() - t1
    timings["total"] = time.time() - t0

    n_target = int(target["track_id"].notna().sum())
    quality = {
        "target_frames": n_target,
        "target_coverage_pct": 100.0 * n_target / max(1, len(target)),
        "occluded_frames": int((target["status"] == "occluded").sum()),
        "linked_frames": int((target["status"] == "linked").sum()),
        "identity_links": sum(1 for e in tres.events if e["type"] == "identity_link"),
        "possible_id_switches": sum(1 for e in tres.events if e["type"] == "possible_id_switch"),
        "tracking_gaps": sum(1 for e in tres.events if e["type"] == "tracking_gap"),
        "people_tracked": int(det["tracks"]["track_id"].nunique()),
        "seed_track": tres.seed_track,
        "selection": {"auto": bool(sel.get("auto", False)), "t": sel["t"]},
        "calibration": cal_info,
    }
    summary = {
        "pipeline_version": PIPELINE_VERSION,
        "synthetic": synthetic,
        "video": meta.to_dict(),
        "analysis": {
            "fps": prof["analysis_fps"],
            "width": aw,
            "height": ah,
            "scale": scale,
            "performance_mode": s.performance_mode,
            "profile": prof,
            "detector": det_name,
            "tracker": s.tracker,
            "cache": {"detections": det["cached"], "pose": pose["cached"]},
        },
        "settings": s.to_dict(),
        "metrics": res["metrics"],
        "quality": quality,
        "timings_s": timings,
        "throughput_fps": len(det["samples"]) / max(timings["detecting"] + timings["pose"], 1e-6),
        "calibration_used": cal.to_dict() if cal is not None else None,
    }
    summary = clean_json(summary)
    (out / "summary.json").write_text(json.dumps(summary, default=_json_default, indent=1))
    progress("RENDERING", 1.0, "done")
    return summary


def clean_json(o):
    """Recursively replace NaN/inf with None and numpy scalars with Python ones (strict JSON)."""
    if isinstance(o, dict):
        return {str(k): clean_json(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean_json(v) for v in o]
    if isinstance(o, (float, np.floating)):
        return float(o) if np.isfinite(o) else None
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, np.ndarray):
        return clean_json(o.tolist())
    if o is pd.NA:
        return None
    return o


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return None if not np.isfinite(o) else float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, np.bool_):
        return bool(o)
    if o is pd.NA:
        return None
    if isinstance(o, Path):
        return str(o)
    if isinstance(o, Detection):
        return {"bbox": o.bbox.tolist(), "score": o.score}
    raise TypeError(f"not serialisable: {type(o)}")
