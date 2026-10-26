"""Benchmark annotations per session and evaluation against them; stroke model registry."""

from __future__ import annotations

import json
from datetime import UTC, datetime

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.config import Settings, get_settings
from app.core.deps import get_session_or_404
from app.db.models import ModelVersion, StrokeEvent, Video, VideoSession
from app.db.session import get_db
from app.services.analysis import results_dir
from app.services.serialize import stroke_out
from app.services.storage import safe_path
from vision.evaluation.benchmark import FORMAT, evaluate_session, truth_to_annotations
from vision.pipeline import clean_json

router = APIRouter(prefix="/api", tags=["evaluation"])


def _ann_path(settings: Settings, s: VideoSession):
    return settings.sessions_dir / s.id / "benchmark" / "annotations.json"


@router.get("/sessions/{session_id}/benchmark", summary="Benchmark annotations of a session")
def get_annotations(
    s: VideoSession = Depends(get_session_or_404),
    settings: Settings = Depends(get_settings),
    db: Session = Depends(get_db),
):
    p = _ann_path(settings, s)
    if p.exists():
        return json.loads(p.read_text())
    video = db.get(Video, s.video_id)
    if video.is_synthetic and video.truth_path:
        return truth_to_annotations(json.loads(safe_path(settings, video.truth_path).read_text()))
    return {
        "format": FORMAT,
        "source": "manual",
        "target_id": "P1",
        "frames": [],
        "calibration_points": [],
        "court_positions": [],
        "strokes": [],
    }


@router.put("/sessions/{session_id}/benchmark", summary="Save benchmark annotations")
def put_annotations(
    doc: dict, s: VideoSession = Depends(get_session_or_404), settings: Settings = Depends(get_settings)
):
    if doc.get("format") != FORMAT:
        raise HTTPException(status_code=422, detail=f"format must be '{FORMAT}'")
    for fr in doc.get("frames", []):
        for p in fr.get("people", []):
            b = p.get("bbox")
            if not (isinstance(b, list) and len(b) == 4 and b[2] > b[0] and b[3] > b[1]):
                raise HTTPException(status_code=422, detail=f"invalid bbox at t={fr.get('t')}")
    doc["updated_at"] = datetime.now(UTC).isoformat()
    p = _ann_path(settings, s)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(doc, indent=1))
    return {"saved": True, "frames": len(doc.get("frames", [])), "strokes": len(doc.get("strokes", []))}


@router.get("/sessions/{session_id}/evaluation", summary="Evaluate the latest analysis against the annotations")
def evaluate(
    s: VideoSession = Depends(get_session_or_404),
    settings: Settings = Depends(get_settings),
    db: Session = Depends(get_db),
):
    rd = results_dir(settings, s)
    if rd is None:
        raise HTTPException(status_code=409, detail="run the analysis first")
    ann = get_annotations(s, settings, db)
    if not (ann.get("frames") or ann.get("strokes") or ann.get("calibration_points") or ann.get("court_positions")):
        raise HTTPException(status_code=409, detail="no annotations yet; add some in the annotation tool")
    strokes = [stroke_out(e) for e in db.scalars(select(StrokeEvent).where(StrokeEvent.session_id == s.id))]
    return clean_json(evaluate_session(rd, ann, strokes))


@router.get("/models", summary="Registered stroke-classifier versions and model cards")
def list_models(db: Session = Depends(get_db)):
    rows = db.scalars(select(ModelVersion).order_by(ModelVersion.created_at.desc())).all()
    return [
        {
            "id": m.id,
            "task": m.task,
            "version": m.version,
            "is_active": m.is_active,
            "metrics": m.metrics,
            "card": m.card,
            "created_at": m.created_at,
        }
        for m in rows
    ]
