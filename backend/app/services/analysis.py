"""Glue between the database and the framework-free vision pipeline."""

from __future__ import annotations

import shutil
import threading
from pathlib import Path

import numpy as np
import pandas as pd
from sqlalchemy import delete, select

from app.core.config import Settings
from app.db.models import (
    AnalysisJob,
    CourtCalibration,
    CourtPositions,
    ModelVersion,
    MovementMetrics,
    PlayerTrack,
    PoseLandmarks,
    StrokeEvent,
    Video,
    VideoSession,
)
from app.db.session import session_factory
from app.services.serialize import active_calibration
from app.services.storage import relative_to_data, safe_path
from vision.pipeline import PipelineInputs, PipelineSettings, run_pipeline

KEEP_DISTANCE_S = 0.25


def results_dir(settings: Settings, session: VideoSession) -> Path | None:
    if not session.results_dir:
        return None
    p = safe_path(settings, session.results_dir)
    return p if p.exists() else None


def _calibration_dict(c: CourtCalibration) -> dict:
    return {
        "homography": c.homography,
        "image_points": c.image_points,
        "court_points": None,
        "point_ids": c.point_ids,
        "inliers": c.inliers,
        "method": c.method,
        "image_size": [c.image_width, c.image_height],
        "errors_m": c.errors_m,
        "errors_px": c.errors_px,
        "condition_number": c.condition_number,
        "warnings": c.warnings,
    }


def calibration_for_pipeline(c: CourtCalibration) -> dict:
    from vision.geometry.court import keypoint_by_id

    d = _calibration_dict(c)
    d["court_points"] = [[keypoint_by_id(k).x, keypoint_by_id(k).y] for k in c.point_ids]
    return d


def active_stroke_model_dir(settings: Settings) -> Path | None:
    with session_factory()() as db:
        mv = db.scalars(select(ModelVersion).where(ModelVersion.is_active.is_(True))).first()
        if mv is None:
            return None
        p = Path(mv.path)
        return p.parent if p.exists() else None


def execute_job(job_id: str, settings: Settings, progress, cancel: threading.Event) -> dict:
    from app.services.demo import ensure_demo_rendered

    with session_factory()() as db:
        job = db.get(AnalysisJob, job_id)
        video_id = db.get(VideoSession, job.session_id).video_id
    ensure_demo_rendered(video_id, settings, progress, cancel)
    with session_factory()() as db:
        job = db.get(AnalysisJob, job_id)
        session = db.get(VideoSession, job.session_id)
        video = db.get(Video, session.video_id)
        cal = active_calibration(session)
        out_dir = settings.sessions_dir / session.id / "results" / job_id
        inputs = PipelineInputs(
            video_path=safe_path(settings, video.stored_path),
            video_hash=video.sha256,
            out_dir=out_dir,
            cache_dir=settings.cache_dir,
            models_dir=settings.models_dir,
            settings=PipelineSettings.from_dict(job.settings),
            calibration=calibration_for_pipeline(cal) if cal else None,
            selection=session.player_selection,
            corrections=list(session.corrections or []),
            synthetic_truth=safe_path(settings, video.truth_path)
            if (video.is_synthetic and video.truth_path)
            else None,
            stroke_model_dir=active_stroke_model_dir(settings),
        )
        needs_proxy = not video.browser_playable and not video.proxy_path
        session_id = session.id

    summary = run_pipeline(inputs, progress, cancel)

    if needs_proxy:
        from vision.video.writer import transcode_for_browser

        progress("RENDERING", 0.5, "creating a browser-playable preview of the video")
        proxy_stem = settings.uploads_dir / summary_video_dir(inputs.video_path) / "proxy"
        path, _codec = transcode_for_browser(inputs.video_path, proxy_stem, cancel=cancel)
        with session_factory()() as db:
            v = db.get(Video, db.get(VideoSession, session_id).video_id)
            v.proxy_path = relative_to_data(settings, path)
            db.commit()

    persist_results(job_id, session_id, out_dir, summary, settings)
    return summary["timings_s"]


def summary_video_dir(video_path: Path) -> str:
    return video_path.parent.name


def persist_results(job_id: str, session_id: str, out_dir: Path, summary: dict, settings: Settings) -> None:
    import json

    events = json.loads((out_dir / "events.json").read_text())
    positions = pd.read_parquet(out_dir / "positions.parquet")
    tracks = pd.read_parquet(out_dir / "tracks.parquet")
    with session_factory()() as db:
        session = db.get(VideoSession, session_id)
        previous = session.results_dir
        session.results_dir = relative_to_data(settings, out_dir)
        session.settings = summary["settings"]
        m = summary["metrics"]
        session.summary = compact_summary(summary)

        if summary.get("quality", {}).get("calibration", {}).get("calibration") and active_calibration(session) is None:
            c = summary["quality"]["calibration"]["calibration"]
            db.add(
                CourtCalibration(
                    session_id=session_id,
                    method="auto",
                    frame_time_s=0.0,
                    image_width=c["image_size"][0],
                    image_height=c["image_size"][1],
                    point_ids=c["point_ids"],
                    image_points=c["image_points"],
                    homography=c["homography"],
                    inliers=c["inliers"],
                    errors_m=c["errors_m"],
                    errors_px=c["errors_px"],
                    mean_error_m=c["mean_error_m"],
                    max_error_m=c["max_error_m"],
                    mean_error_px=c["mean_error_px"],
                    condition_number=c["condition_number"],
                    warnings=c["warnings"],
                    is_active=True,
                )
            )

        db.execute(delete(PlayerTrack).where(PlayerTrack.session_id == session_id))
        target_ids = set(positions["track_id"].dropna().astype(int).tolist())
        for tid, g in tracks.groupby("track_id"):
            db.add(
                PlayerTrack(
                    session_id=session_id,
                    job_id=job_id,
                    track_id=int(tid),
                    first_t=float(g["t"].min()),
                    last_t=float(g["t"].max()),
                    n_frames=len(g),
                    mean_score=float(g["score"].mean()),
                    is_target=int(tid) in target_ids,
                )
            )
        db.execute(delete(PoseLandmarks).where(PoseLandmarks.session_id == session_id))
        db.add(
            PoseLandmarks(
                session_id=session_id,
                job_id=job_id,
                artifact_path=relative_to_data(settings, out_dir / "pose.npz"),
                model_name=summary["analysis"]["profile"]["pose_model"],
                frames_with_pose=int(m.get("pose", {}).get("frames_with_pose", 0)),
            )
        )
        db.execute(delete(CourtPositions).where(CourtPositions.session_id == session_id))
        has_pos = positions["court_x"].notna() if "court_x" in positions else pd.Series(False, index=positions.index)
        db.add(
            CourtPositions(
                session_id=session_id,
                job_id=job_id,
                artifact_path=relative_to_data(settings, out_dir / "positions.parquet"),
                n_samples=len(positions),
                n_with_position=int(has_pos.sum()),
                n_interpolated=int(positions.get("interpolated", pd.Series(False)).fillna(False).astype(bool).sum()),
            )
        )
        db.add(
            MovementMetrics(
                session_id=session_id, job_id=job_id, pipeline_version=summary["pipeline_version"], metrics=m
            )
        )

        # Strokes: keep manual events and anything the user reviewed or labelled; replace the rest.
        existing = db.scalars(select(StrokeEvent).where(StrokeEvent.session_id == session_id)).all()
        keep = [
            e
            for e in existing
            if e.source == "manual" or e.review_status != "candidate" or e.label_shot_type or e.label_stroke_side
        ]
        for e in existing:
            if e not in keep:
                db.delete(e)
        kept_t = np.array([e.t for e in keep])
        for ev in events:
            if ev["type"] != "stroke":
                continue
            if kept_t.size and np.min(np.abs(kept_t - ev["t"])) < KEEP_DISTANCE_S:
                continue
            db.add(
                StrokeEvent(
                    session_id=session_id,
                    job_id=job_id,
                    t=ev["t"],
                    t_start=ev.get("t_start"),
                    t_end=ev.get("t_end"),
                    source="detected",
                    confidence=ev.get("confidence"),
                    shot_type=ev.get("shot_type"),
                    shot_type_confidence=ev.get("shot_type_confidence"),
                    stroke_side=ev.get("stroke_side"),
                    stroke_side_confidence=ev.get("stroke_side_confidence"),
                    height_class=ev.get("height_class"),
                    classifier=ev.get("classifier"),
                    signals=ev.get("signals"),
                )
            )
        db.commit()
    if previous and previous != relative_to_data(settings, out_dir):
        old = safe_path(settings, previous)
        if old.exists() and old.parent.name == "results":
            shutil.rmtree(old, ignore_errors=True)


def compact_summary(summary: dict) -> dict:
    """The part of the summary kept on the session row for fast dashboard listing."""
    m = summary["metrics"]
    q = summary["quality"]
    out = {
        "synthetic": summary["synthetic"],
        "calibrated": m.get("calibrated", True) is not False,
        "duration_s": summary["video"]["duration_s"],
        "pipeline_version": summary["pipeline_version"],
        "performance_mode": summary["analysis"]["performance_mode"],
        "detector": summary["analysis"]["detector"],
        "target_coverage_pct": q["target_coverage_pct"],
        "strokes": m.get("strokes", {}).get("candidates"),
        "calibration_error_m": (q.get("calibration") or {}).get("mean_error_m"),
        "throughput_fps": summary.get("throughput_fps"),
    }
    if out["calibrated"]:
        out.update(
            {
                "distance_m": m["distance"]["total_m"],
                "distance_per_min_m": m["distance"]["per_minute_m"],
                "mean_speed_mps": m["speed"]["mean_mps"],
                "max_speed_mps": m["speed"]["max_filtered_mps"],
                "p95_speed_mps": m["speed"]["p95_mps"],
                "zones_visited_pct": m["zones"]["visited_pct"],
                "direction_changes": m["direction_changes"],
                "observed_s": m["observed_s"],
                "recovery_time_median_s": m["recovery"]["recovery_time_median_s"],
                "composite_score": (m.get("composite") or {}).get("score"),
            }
        )
    return out
