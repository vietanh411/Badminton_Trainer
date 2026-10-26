"""Health, capabilities, court geometry and session comparison."""

from __future__ import annotations

import platform

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.core.config import Settings, get_settings
from app.db.models import VideoSession
from app.db.session import get_db
from app.services import results as R
from app.services.analysis import results_dir
from app.services.compare import compare
from vision.geometry.court import court_geometry_payload
from vision.pipeline import MEDIAPIPE_DETECTOR, PIPELINE_VERSION, PROFILES

router = APIRouter(prefix="/api", tags=["system"])
public_router = APIRouter(prefix="/api", tags=["system"])


@public_router.get("/health", summary="Liveness probe (never requires the API key)")
def health():
    return {"status": "ok", "pipeline_version": PIPELINE_VERSION}


def _device_info() -> dict:
    info = {"device": "cpu", "gpu": None, "torch": None, "ultralytics": None}
    try:
        import torch

        info["torch"] = torch.__version__
        if torch.cuda.is_available():
            info["device"] = "cuda"
            info["gpu"] = torch.cuda.get_device_name(0)
    except ImportError:
        pass
    try:
        import ultralytics

        info["ultralytics"] = ultralytics.__version__
    except ImportError:
        pass
    return info


@router.get("/system", summary="Capabilities: models present, device, limits, performance profiles")
def system(settings: Settings = Depends(get_settings)):
    import cv2
    import mediapipe

    models = {}
    names = (
        {p["yolo_weights"] for p in PROFILES.values()}
        | {p["pose_model"] for p in PROFILES.values()}
        | {MEDIAPIPE_DETECTOR}
    )
    for name in sorted(names):
        models[name] = (settings.models_dir / name).exists()
    return {
        "pipeline_version": PIPELINE_VERSION,
        "python": platform.python_version(),
        "opencv": cv2.__version__,
        "mediapipe": mediapipe.__version__,
        **_device_info(),
        "models": models,
        "profiles": PROFILES,
        "limits": {
            "max_upload_mb": settings.max_upload_mb,
            "max_duration_s": settings.max_duration_s,
            "max_concurrent_jobs": settings.max_concurrent_jobs,
            "max_queued_jobs": settings.max_queued_jobs,
            "allowed_extensions": list(settings.allowed_extensions),
        },
        "auth_required": bool(settings.api_key),
        "demo_enabled": settings.enable_demo,
    }


@router.get("/court", summary="Court model: lines and calibration keypoints in metres")
def court():
    return court_geometry_payload()


@router.get("/compare", summary="Compare two analysed sessions with normalised metrics")
def compare_sessions(a: str, b: str, db: Session = Depends(get_db), settings: Settings = Depends(get_settings)):
    out = {}
    for key, sid in (("a", a), ("b", b)):
        s = db.get(VideoSession, sid)
        if s is None:
            raise HTTPException(status_code=404, detail=f"session {sid} not found")
        try:
            res = R.load_results(results_dir(settings, s))
        except R.NoResults as exc:
            raise HTTPException(status_code=409, detail=f"session '{s.name}' has no completed analysis") from exc
        out[key] = (s, res)
    sa, ra = out["a"]
    sb, rb = out["b"]
    result = compare(ra["summary"], rb["summary"])
    result["a"] = {"id": sa.id, "name": sa.name, "synthetic": ra["summary"]["synthetic"]}
    result["b"] = {"id": sb.id, "name": sb.name, "synthetic": rb["summary"]["synthetic"]}
    return result
