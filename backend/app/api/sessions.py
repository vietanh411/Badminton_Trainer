"""Session workflow: calibration, player selection, analysis, results, strokes, exports."""

from __future__ import annotations

import asyncio
import csv
import io
import json

import numpy as np
from fastapi import APIRouter, Depends, HTTPException, Query, Response
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.config import Settings, get_settings
from app.core.deps import get_jobs, get_session_or_404
from app.db.models import (
    AnalysisJob,
    CourtCalibration,
    ExportArtifact,
    PlayerTrack,
    StrokeEvent,
    Video,
    VideoSession,
)
from app.db.session import get_db
from app.schemas.api import (
    AnalyzeIn,
    CalibrationIn,
    CorrectionIn,
    PlayerSelectionIn,
    SessionPatch,
    StrokeIn,
    StrokePatch,
)
from app.services import results as R
from app.services.analysis import results_dir
from app.services.report import build_report
from app.services.serialize import calibration_out, calibration_quality, job_out, session_out, stroke_out
from app.services.storage import remove_tree, safe_path
from app.workers.jobs import ACTIVE, JobManager, QueueFull
from vision.calibration.auto import auto_calibrate
from vision.calibration.homography import CalibrationError, fit_homography
from vision.pipeline import PROFILES, PipelineSettings, clean_json, resolve_detector_name
from vision.video.reader import Frame, VideoError, read_frame_at

router = APIRouter(prefix="/api/sessions", tags=["sessions"])


def _latest_job(db: Session, s: VideoSession) -> AnalysisJob | None:
    return db.get(AnalysisJob, s.latest_job_id) if s.latest_job_id else None


@router.get("", summary="List sessions (newest first)")
def list_sessions(db: Session = Depends(get_db)):
    rows = db.scalars(select(VideoSession).order_by(VideoSession.created_at.desc())).all()
    return [session_out(s, latest_job=_latest_job(db, s)) for s in rows]


@router.get("/{session_id}", summary="Session details")
def get_session(s: VideoSession = Depends(get_session_or_404), db: Session = Depends(get_db)):
    return session_out(s, latest_job=_latest_job(db, s))


@router.patch("/{session_id}", summary="Rename a session or edit notes")
def patch_session(body: SessionPatch, s: VideoSession = Depends(get_session_or_404), db: Session = Depends(get_db)):
    if body.name is not None:
        s.name = body.name
    if body.notes is not None:
        s.notes = body.notes
    db.commit()
    return session_out(s, latest_job=_latest_job(db, s))


@router.delete("/{session_id}", status_code=204, summary="Delete a session, its artifacts and (if unused) its video")
def delete_session(
    s: VideoSession = Depends(get_session_or_404),
    db: Session = Depends(get_db),
    settings: Settings = Depends(get_settings),
    jobs: JobManager = Depends(get_jobs),
):
    running = db.scalars(
        select(AnalysisJob).where(AnalysisJob.session_id == s.id, AnalysisJob.status.in_(ACTIVE))
    ).first()
    if running:
        raise HTTPException(status_code=409, detail="cancel the running analysis before deleting this session")
    video = db.get(Video, s.video_id)
    remove_tree(settings.sessions_dir / s.id, settings)
    db.delete(s)
    db.flush()
    others = db.scalars(select(VideoSession).where(VideoSession.video_id == video.id)).first()
    if others is None:
        remove_tree(settings.uploads_dir / video.id, settings)
        db.delete(video)
    db.commit()
    return Response(status_code=204)


# ------------------------------------------------------------------------------ calibration


@router.get("/{session_id}/calibration", summary="Active court calibration")
def get_calibration(s: VideoSession = Depends(get_session_or_404)):
    from app.services.serialize import active_calibration

    return calibration_out(active_calibration(s))


@router.post("/{session_id}/calibration", summary="Fit, validate and (unless dry_run) save a manual calibration")
def post_calibration(body: CalibrationIn, s: VideoSession = Depends(get_session_or_404), db: Session = Depends(get_db)):
    video = db.get(Video, s.video_id)
    sx, sy = video.width / body.image_width, video.height / body.image_height
    pts = np.array([[p.x * sx, p.y * sy] for p in body.points])
    try:
        cal = fit_homography(
            pts, point_ids=[p.id for p in body.points], method="manual", image_size=(video.width, video.height)
        )
    except CalibrationError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    d = cal.to_dict()
    out = {
        "id": None,
        "method": "manual",
        "frame_time_s": body.frame_time_s,
        "image_width": video.width,
        "image_height": video.height,
        "point_ids": d["point_ids"],
        "image_points": d["image_points"],
        "homography": d["homography"],
        "inliers": d["inliers"],
        "errors_m": d["errors_m"],
        "errors_px": d["errors_px"],
        "mean_error_m": d["mean_error_m"],
        "max_error_m": d["max_error_m"],
        "mean_error_px": d["mean_error_px"],
        "condition_number": d["condition_number"],
        "warnings": d["warnings"],
        "quality": calibration_quality(d["mean_error_m"], int(sum(d["inliers"])), d["warnings"]),
        "saved": False,
    }
    if body.dry_run:
        return clean_json(out)
    for c in s.calibrations:
        c.is_active = False
    row = CourtCalibration(
        session_id=s.id,
        method="manual",
        frame_time_s=body.frame_time_s,
        image_width=video.width,
        image_height=video.height,
        point_ids=d["point_ids"],
        image_points=d["image_points"],
        homography=d["homography"],
        inliers=d["inliers"],
        errors_m=d["errors_m"],
        errors_px=d["errors_px"],
        mean_error_m=d["mean_error_m"],
        max_error_m=d["max_error_m"],
        mean_error_px=d["mean_error_px"],
        condition_number=d["condition_number"],
        warnings=d["warnings"],
        is_active=True,
    )
    db.add(row)
    db.commit()
    return clean_json(calibration_out(row))


@router.delete("/{session_id}/calibration", status_code=204, summary="Reset calibration (deactivate all)")
def reset_calibration(s: VideoSession = Depends(get_session_or_404), db: Session = Depends(get_db)):
    for c in s.calibrations:
        c.is_active = False
    db.commit()
    return Response(status_code=204)


@router.post("/{session_id}/calibration/auto", summary="Suggest a calibration by detecting court lines in a frame")
async def auto_calibration(
    t: float = Query(1.0, ge=0),
    s: VideoSession = Depends(get_session_or_404),
    db: Session = Depends(get_db),
    settings: Settings = Depends(get_settings),
):
    video = db.get(Video, s.video_id)
    try:
        img, actual = await asyncio.to_thread(read_frame_at, safe_path(settings, video.stored_path), t)
    except VideoError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    res = await asyncio.to_thread(auto_calibrate, img)
    if not res.success:
        return {"success": False, "message": res.message, "score": res.score}
    d = res.calibration.to_dict()
    return clean_json(
        {
            "success": True,
            "message": "court detected — review the points before saving",
            "score": res.score,
            "frame_time_s": actual,
            "image_width": img.shape[1],
            "image_height": img.shape[0],
            "points": [{"id": i, "x": p[0], "y": p[1]} for i, p in zip(d["point_ids"], d["image_points"], strict=True)],
            "mean_error_m": d["mean_error_m"],
            "warnings": d["warnings"],
        }
    )


# ------------------------------------------------------------------------- player selection


@router.get("/{session_id}/detections", summary="Detect people in one frame for player selection")
async def detect_people(
    t: float = Query(1.0, ge=0),
    s: VideoSession = Depends(get_session_or_404),
    db: Session = Depends(get_db),
    settings: Settings = Depends(get_settings),
):
    from vision.detection.base import filter_detections
    from vision.pipeline import build_detector

    video = db.get(Video, s.video_id)
    path = safe_path(settings, video.stored_path)
    synthetic = video.is_synthetic and video.truth_path is not None
    ps = PipelineSettings.from_dict(s.settings or {})
    name = resolve_detector_name(ps, settings.models_dir, synthetic)

    def run():
        img, actual = read_frame_at(path, t)
        det = build_detector(
            name, ps, settings.models_dir, safe_path(settings, video.truth_path) if synthetic else None
        )
        try:
            frame = Frame(index=0, sample_index=0, timestamp_s=actual, image=img, scale=1.0)
            dets = filter_detections(det.detect(frame), frame_height=img.shape[0])
        finally:
            det.close()
        return actual, dets, img.shape

    try:
        actual, dets, shape = await asyncio.to_thread(run)
    except VideoError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:  # model missing etc.
        raise HTTPException(status_code=503, detail=f"person detector unavailable: {exc}") from exc
    dets = sorted(dets, key=lambda d: -d.score)
    return {
        "t": actual,
        "detector": name,
        "image_width": shape[1],
        "image_height": shape[0],
        "people": [{"bbox": [round(float(v), 1) for v in d.bbox], "score": round(d.score, 3)} for d in dets],
    }


@router.post("/{session_id}/player-selection", summary="Save the target player (box on a frame)")
def select_player(
    body: PlayerSelectionIn, s: VideoSession = Depends(get_session_or_404), db: Session = Depends(get_db)
):
    s.player_selection = {"t": body.t, "bbox": body.bbox}
    s.corrections = []
    db.commit()
    return session_out(s, latest_job=_latest_job(db, s))


@router.get("/{session_id}/tracks", summary="Tracker identities from the latest analysis (for corrections)")
def list_tracks(s: VideoSession = Depends(get_session_or_404), db: Session = Depends(get_db)):
    rows = db.scalars(select(PlayerTrack).where(PlayerTrack.session_id == s.id).order_by(PlayerTrack.first_t)).all()
    return [
        {
            "track_id": r.track_id,
            "first_t": r.first_t,
            "last_t": r.last_t,
            "n_frames": r.n_frames,
            "mean_score": r.mean_score,
            "is_target": r.is_target,
        }
        for r in rows
    ]


@router.post("/{session_id}/corrections", summary="Correct the target identity from time t (re-run analysis to apply)")
def add_correction(body: CorrectionIn, s: VideoSession = Depends(get_session_or_404), db: Session = Depends(get_db)):
    corr = [c for c in (s.corrections or []) if abs(c["t"] - body.t) > 1e-3]
    corr.append({"t": body.t, "track_id": body.track_id})
    s.corrections = sorted(corr, key=lambda c: c["t"])
    db.commit()
    return {"corrections": s.corrections}


@router.delete("/{session_id}/corrections", summary="Remove all identity corrections")
def clear_corrections(s: VideoSession = Depends(get_session_or_404), db: Session = Depends(get_db)):
    s.corrections = []
    db.commit()
    return {"corrections": []}


# --------------------------------------------------------------------------------- analysis


@router.post("/{session_id}/analyze", status_code=202, summary="Start (or re-run) the analysis job")
def analyze(
    body: AnalyzeIn | None = None,
    s: VideoSession = Depends(get_session_or_404),
    db: Session = Depends(get_db),
    jobs: JobManager = Depends(get_jobs),
):
    current = PipelineSettings.from_dict(s.settings or {}).to_dict()
    patch = body.model_dump(exclude_none=True) if body else {}
    for k, v in patch.items():
        current[k] = {**current[k], **v} if isinstance(v, dict) and isinstance(current.get(k), dict) else v
    video = db.get(Video, s.video_id)
    if not video.is_synthetic and current.get("detector") == "truth":
        raise HTTPException(status_code=422, detail="the ground-truth detector is only available for synthetic videos")
    try:
        settings_obj = PipelineSettings.from_dict(current)
    except (TypeError, ValueError) as exc:
        raise HTTPException(status_code=422, detail=f"invalid settings: {exc}") from exc
    s.settings = settings_obj.to_dict()
    db.commit()
    try:
        job_id = jobs.submit(s.id, settings_obj.to_dict())
    except QueueFull as exc:
        raise HTTPException(status_code=429, detail=str(exc)) from exc
    return job_out(db.get(AnalysisJob, job_id))


def _results(s: VideoSession, settings: Settings) -> dict:
    try:
        return R.load_results(results_dir(settings, s))
    except R.NoResults as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.get("/{session_id}/metrics", summary="All movement, pose, stroke and quality metrics of the latest analysis")
def metrics(s: VideoSession = Depends(get_session_or_404), settings: Settings = Depends(get_settings)):
    res = _results(s, settings)
    sm = res["summary"]
    return {
        "synthetic": sm["synthetic"],
        "metrics": sm["metrics"],
        "quality": sm["quality"],
        "analysis": sm["analysis"],
        "video": sm["video"],
        "timings_s": sm["timings_s"],
        "pipeline_version": sm["pipeline_version"],
        "settings": sm["settings"],
        "calibration": sm["calibration_used"],
        "profiles": PROFILES,
    }


@router.get("/{session_id}/trajectory", summary="Raw and filtered court trajectory")
def trajectory(
    start: float | None = None,
    end: float | None = None,
    max_points: int = Query(5000, ge=10, le=50000),
    s: VideoSession = Depends(get_session_or_404),
    settings: Settings = Depends(get_settings),
):
    return R.trajectory_payload(_results(s, settings), start, end, max_points)


@router.get("/{session_id}/heatmap", summary="Time-weighted occupancy heatmap (JSON grid, PNG or SVG)")
def heatmap(
    start: float | None = None,
    end: float | None = None,
    preset: str | None = Query(None, pattern="^(full|first_half|second_half)$"),
    zone: str | None = Query(None, pattern="^(front|mid|rear)$"),
    cell_m: float | None = Query(None, ge=0.05, le=1.0),
    sigma_m: float | None = Query(None, ge=0, le=2.0),
    normalize: str = Query("share", pattern="^(share|seconds|density)$"),
    format: str = Query("json", pattern="^(json|png|svg)$"),
    points: bool = False,
    trajectory: bool = False,
    opacity: float = Query(0.9, ge=0.1, le=1.0),
    s: VideoSession = Depends(get_session_or_404),
    settings: Settings = Depends(get_settings),
):
    res = _results(s, settings)
    if preset in ("first_half", "second_half"):
        t = res["positions"]["t"]
        mid = float(t.min() + (t.max() - t.min()) / 2)
        start, end = (float(t.min()), mid) if preset == "first_half" else (mid, float(t.max()))
    payload = R.heatmap_payload(
        res, start=start, end=end, zone=zone, cell_m=cell_m, sigma_m=sigma_m, normalize=normalize
    )
    if format == "json" or not payload.get("available"):
        if format != "json":
            raise HTTPException(status_code=409, detail=payload.get("reason"))
        return payload
    from vision.visualization.court_plot import figure_bytes, heatmap_figure

    hm, pts = R.heatmap_object(res, start=start, end=end, sigma_m=sigma_m)
    title = "Court occupancy" + (" (SYNTHETIC DEMO)" if res["summary"]["synthetic"] else "")
    fig = heatmap_figure(
        hm,
        title=title,
        points=pts if points else None,
        trajectory=pts if trajectory else None,
        opacity=opacity,
        normalize=normalize,
    )
    data = figure_bytes(fig, format)
    return Response(
        data,
        media_type="image/png" if format == "png" else "image/svg+xml",
        headers={"Content-Disposition": f'attachment; filename="heatmap_{s.id[:8]}.{format}"'},
    )


@router.get("/{session_id}/events", summary="Timeline events (strokes, direction changes, tracking problems...)")
def events(
    types: str | None = None,
    s: VideoSession = Depends(get_session_or_404),
    db: Session = Depends(get_db),
    settings: Settings = Depends(get_settings),
):
    res = _results(s, settings)
    wanted = set(types.split(",")) if types else None
    evs = [e for e in res["events"] if e["type"] != "stroke" and (wanted is None or e["type"] in wanted)]
    if wanted is None or "stroke" in wanted:
        strokes = db.scalars(select(StrokeEvent).where(StrokeEvent.session_id == s.id).order_by(StrokeEvent.t)).all()
        evs += [{"type": "stroke", **stroke_out(e)} for e in strokes]
    return sorted(evs, key=lambda e: e["t"])


@router.get("/{session_id}/annotations", summary="Overlay data (boxes, pose, positions) for a playback window")
def annotations(
    start: float = Query(0, ge=0),
    end: float = Query(30, ge=0),
    s: VideoSession = Depends(get_session_or_404),
    settings: Settings = Depends(get_settings),
):
    if end - start > 120:
        raise HTTPException(status_code=422, detail="request at most 120 s of annotations at a time")
    return R.annotations_payload(_results(s, settings), start, end)


# ---------------------------------------------------------------------------------- strokes


@router.get("/{session_id}/strokes", summary="Stroke events with review status and labels")
def list_strokes(s: VideoSession = Depends(get_session_or_404), db: Session = Depends(get_db)):
    rows = db.scalars(select(StrokeEvent).where(StrokeEvent.session_id == s.id).order_by(StrokeEvent.t)).all()
    return [stroke_out(e) for e in rows]


@router.post("/{session_id}/strokes", status_code=201, summary="Add a manually annotated stroke")
def add_stroke(body: StrokeIn, s: VideoSession = Depends(get_session_or_404), db: Session = Depends(get_db)):
    e = StrokeEvent(
        session_id=s.id,
        t=body.t,
        source="manual",
        review_status="confirmed",
        label_shot_type=body.label_shot_type,
        label_stroke_side=body.label_stroke_side,
    )
    db.add(e)
    db.commit()
    return stroke_out(e)


stroke_router = APIRouter(prefix="/api/strokes", tags=["strokes"])


@stroke_router.patch("/{stroke_id}", summary="Review / label / move a stroke event")
def patch_stroke(stroke_id: str, body: StrokePatch, db: Session = Depends(get_db)):
    e = db.get(StrokeEvent, stroke_id)
    if e is None:
        raise HTTPException(status_code=404, detail="stroke not found")
    data = body.model_dump(exclude_unset=True)
    if data.pop("clear_labels", False):
        e.label_shot_type = None
        e.label_stroke_side = None
    for k, v in data.items():
        setattr(e, k, v)
    db.commit()
    return stroke_out(e)


@stroke_router.delete("/{stroke_id}", status_code=204, summary="Delete a stroke event")
def delete_stroke(stroke_id: str, db: Session = Depends(get_db)):
    e = db.get(StrokeEvent, stroke_id)
    if e is None:
        raise HTTPException(status_code=404, detail="stroke not found")
    db.delete(e)
    db.commit()
    return Response(status_code=204)


# ---------------------------------------------------------------------------------- exports


def _record_export(db: Session, s: VideoSession, kind: str, size: int) -> None:
    db.add(ExportArtifact(session_id=s.id, kind=kind, path="(streamed)", size_bytes=size))
    db.commit()


@router.get("/{session_id}/report", summary="PDF analysis report")
def report(
    s: VideoSession = Depends(get_session_or_404),
    db: Session = Depends(get_db),
    settings: Settings = Depends(get_settings),
):
    res = _results(s, settings)
    strokes = [
        stroke_out(e)
        for e in db.scalars(select(StrokeEvent).where(StrokeEvent.session_id == s.id).order_by(StrokeEvent.t))
    ]
    pdf = build_report({"name": s.name, "strokes": strokes}, res)
    _record_export(db, s, "report.pdf", len(pdf))
    return Response(
        pdf,
        media_type="application/pdf",
        headers={"Content-Disposition": f'attachment; filename="badmintonvision_{s.id[:8]}.pdf"'},
    )


@router.get(
    "/{session_id}/export/{kind}",
    summary="Download data: positions.csv, events.csv, strokes.csv, metrics.json, analysis.json",
)
def export(
    kind: str,
    s: VideoSession = Depends(get_session_or_404),
    db: Session = Depends(get_db),
    settings: Settings = Depends(get_settings),
):
    res = _results(s, settings)
    synthetic_note = "# SYNTHETIC DEMO DATA - not a real player's performance\n" if res["summary"]["synthetic"] else ""
    if kind == "positions.csv":
        pos = res["positions"]
        cols = [
            c
            for c in [
                "t",
                "frame_index",
                "track_id",
                "status",
                "position_source",
                "position_confidence",
                "img_x",
                "img_y",
                "court_x_raw",
                "court_y_raw",
                "raw_valid",
                "reject_reason",
                "court_x",
                "court_y",
                "interpolated",
                "segment",
                "speed",
                "tangential_accel",
                "zone",
                "moving",
                "side",
                "distance_from_base_m",
            ]
            if c in pos
        ]
        body = synthetic_note + pos[cols].to_csv(index=False, float_format="%.4f")
        media = "text/csv"
    elif kind == "events.csv":
        evs = [e for e in res["events"] if e["type"] != "stroke"]
        buf = io.StringIO()
        w = csv.writer(buf)
        w.writerow(["type", "t", "t_end", "details"])
        for e in evs:
            w.writerow(
                [
                    e["type"],
                    e["t"],
                    e.get("t_end", ""),
                    json.dumps({k: v for k, v in e.items() if k not in ("type", "t", "t_end")}),
                ]
            )
        body = synthetic_note + buf.getvalue()
        media = "text/csv"
    elif kind == "strokes.csv":
        rows = [
            stroke_out(e)
            for e in db.scalars(select(StrokeEvent).where(StrokeEvent.session_id == s.id).order_by(StrokeEvent.t))
        ]
        buf = io.StringIO()
        fields = [
            "t",
            "t_start",
            "t_end",
            "source",
            "review_status",
            "confidence",
            "height_class",
            "stroke_side",
            "shot_type",
            "classifier",
            "label_shot_type",
            "label_stroke_side",
        ]
        w = csv.DictWriter(buf, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
        body = synthetic_note + buf.getvalue()
        media = "text/csv"
    elif kind == "metrics.json":
        body = json.dumps(
            clean_json(
                {
                    "synthetic": res["summary"]["synthetic"],
                    "metrics": res["summary"]["metrics"],
                    "quality": res["summary"]["quality"],
                }
            ),
            indent=1,
        )
        media = "application/json"
    elif kind == "analysis.json":
        strokes = [
            stroke_out(e)
            for e in db.scalars(select(StrokeEvent).where(StrokeEvent.session_id == s.id).order_by(StrokeEvent.t))
        ]
        body = json.dumps(
            clean_json({**res["summary"], "events": res["events"], "strokes": strokes}), default=str, indent=1
        )
        media = "application/json"
    else:
        raise HTTPException(status_code=404, detail="unknown export kind")
    data = body.encode()
    _record_export(db, s, kind, len(data))
    return Response(
        data,
        media_type=media,
        headers={"Content-Disposition": f'attachment; filename="badmintonvision_{s.id[:8]}_{kind}"'},
    )
