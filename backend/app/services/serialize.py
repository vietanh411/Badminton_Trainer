"""Model → JSON helpers shared by the API routes."""

from __future__ import annotations

from app.db.models import AnalysisJob, CourtCalibration, StrokeEvent, Video, VideoSession


def calibration_quality(mean_error_m: float, n_points: int, warnings: list[str]) -> str:
    if n_points <= 4:
        return "fair"  # exact fit: error cannot be verified
    if mean_error_m <= 0.08:
        return "good"
    if mean_error_m <= 0.15:
        return "fair"
    return "poor"


def video_out(v: Video) -> dict:
    return {
        "id": v.id,
        "original_filename": v.original_filename,
        "size_bytes": v.size_bytes,
        "container": v.container,
        "codec": v.codec,
        "width": v.width,
        "height": v.height,
        "fps": v.fps,
        "frame_count": v.frame_count,
        "duration_s": v.duration_s,
        "rotation": v.rotation,
        "browser_playable": v.browser_playable,
        "has_proxy": bool(v.proxy_path),
        "is_synthetic": v.is_synthetic,
        "created_at": v.created_at,
        "preview_url": f"/api/videos/{v.id}/preview",
        "thumbnail_url": f"/api/videos/{v.id}/thumbnail",
    }


def calibration_out(c: CourtCalibration | None) -> dict | None:
    if c is None:
        return None
    return {
        "id": c.id,
        "method": c.method,
        "frame_time_s": c.frame_time_s,
        "image_width": c.image_width,
        "image_height": c.image_height,
        "point_ids": c.point_ids,
        "image_points": c.image_points,
        "homography": c.homography,
        "inliers": c.inliers,
        "errors_m": c.errors_m,
        "errors_px": c.errors_px,
        "mean_error_m": c.mean_error_m,
        "max_error_m": c.max_error_m,
        "mean_error_px": c.mean_error_px,
        "condition_number": c.condition_number,
        "warnings": c.warnings,
        "quality": calibration_quality(c.mean_error_m, sum(bool(i) for i in c.inliers), c.warnings),
        "saved": True,
    }


def job_out(j: AnalysisJob | None) -> dict | None:
    if j is None:
        return None
    return {
        "id": j.id,
        "session_id": j.session_id,
        "status": j.status,
        "stage": j.stage,
        "progress": j.progress,
        "message": j.message,
        "error": j.error,
        "created_at": j.created_at,
        "started_at": j.started_at,
        "finished_at": j.finished_at,
        "timings": j.timings,
        "settings": j.settings,
    }


def active_calibration(s: VideoSession) -> CourtCalibration | None:
    active = [c for c in s.calibrations if c.is_active]
    return max(active, key=lambda c: c.created_at) if active else None


def session_out(s: VideoSession, *, include_video: bool = True, latest_job: AnalysisJob | None = None) -> dict:
    summary = s.summary or {}
    return {
        "id": s.id,
        "name": s.name,
        "notes": s.notes,
        "status": s.status,
        "data_source": s.data_source,
        "is_demo": s.is_demo,
        "created_at": s.created_at,
        "updated_at": s.updated_at,
        "video_id": s.video_id,
        "video": video_out(s.video) if include_video else None,
        "settings": s.settings,
        "player_selection": s.player_selection,
        "corrections": s.corrections,
        "calibration": calibration_out(active_calibration(s)),
        "latest_job": job_out(latest_job),
        "summary": summary,
    }


def stroke_out(e: StrokeEvent) -> dict:
    return {
        "id": e.id,
        "t": e.t,
        "t_start": e.t_start,
        "t_end": e.t_end,
        "source": e.source,
        "review_status": e.review_status,
        "confidence": e.confidence,
        "shot_type": e.shot_type,
        "shot_type_confidence": e.shot_type_confidence,
        "stroke_side": e.stroke_side,
        "stroke_side_confidence": e.stroke_side_confidence,
        "height_class": e.height_class,
        "classifier": e.classifier,
        "signals": e.signals,
        "label_shot_type": e.label_shot_type,
        "label_stroke_side": e.label_stroke_side,
        "experimental": True,
    }
