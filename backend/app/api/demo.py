"""Clearly-labelled synthetic demo sessions.

The demo renders a simulated rally video with the synthetic scene generator and analyses it
with ground-truth "detectors". Everything about these sessions is flagged ``synthetic`` in the
database, API responses, exports and the UI. They exist so the product can be explored without
real footage; their metrics describe a simulation, not an athlete.
"""

from __future__ import annotations

import asyncio
import uuid

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from app.api.videos import register_video
from app.core.config import Settings, get_settings
from app.core.deps import get_jobs
from app.db.models import CourtCalibration, Video
from app.db.session import get_db
from app.services.serialize import job_out, session_out
from app.services.storage import file_sha256
from app.workers.jobs import JobManager, QueueFull
from vision.calibration.homography import fit_homography
from vision.pipeline import PipelineSettings
from vision.synthetic.scene import court_keypoint_pixels, make_scene, write_scene

router = APIRouter(prefix="/api/demo", tags=["demo"])

VARIANTS = {
    1: {"seed": 11, "intensity": 1.0, "name": "Demo · synthetic rally A"},
    2: {"seed": 23, "intensity": 1.25, "name": "Demo · synthetic rally B (faster)"},
}
CAL_POINTS = [
    "near_baseline__left_doubles",
    "near_baseline__right_doubles",
    "far_baseline__left_doubles",
    "far_baseline__right_doubles",
    "near_short_service__left_singles",
    "near_short_service__right_singles",
    "far_short_service__left_singles",
    "far_short_service__right_singles",
]


@router.post("", status_code=201, summary="Create and analyse a synthetic demo session")
async def create_demo(
    variant: int = Query(1, ge=1, le=2),
    duration_s: float = Query(45.0, ge=10, le=120),
    db: Session = Depends(get_db),
    settings: Settings = Depends(get_settings),
    jobs: JobManager = Depends(get_jobs),
):
    if not settings.enable_demo:
        raise HTTPException(status_code=404, detail="demo mode is disabled")
    v = VARIANTS[variant]
    settings.tmp_dir.mkdir(parents=True, exist_ok=True)
    stem = settings.tmp_dir / f"demo_{uuid.uuid4().hex}"

    def build():
        scene = make_scene(
            duration=duration_s, fps=30.0, width=1280, height=720, seed=v["seed"], intensity=v["intensity"]
        )
        path, truth = write_scene(scene, stem)
        return scene, path, truth

    scene, path, truth = await asyncio.to_thread(build)
    video, session = await asyncio.to_thread(
        register_video,
        db,
        settings,
        path,
        file_sha256(path),
        path.stat().st_size,
        f"synthetic_demo_{variant}{path.suffix}",
        is_synthetic=True,
        truth=truth,
        session_name=v["name"],
        is_demo=True,
    )
    # The simulator knows the exact camera, so the demo calibration uses exact landmark pixels.
    cal = fit_homography(
        court_keypoint_pixels(scene.camera, CAL_POINTS),
        point_ids=CAL_POINTS,
        method="manual",
        image_size=(video.width, video.height),
    )
    d = cal.to_dict()
    db.add(
        CourtCalibration(
            session_id=session.id,
            method="manual",
            frame_time_s=1.0,
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
            warnings=["synthetic demo: landmarks taken from the simulator"],
            is_active=True,
        )
    )
    import json

    from app.services.storage import safe_path

    tr = json.loads(safe_path(settings, db.get(Video, video.id).truth_path).read_text())
    f = tr["frames"][30]["players"][0]
    session.player_selection = {"t": tr["frames"][30]["t"], "bbox": f["bbox"]}
    ps = PipelineSettings(performance_mode="balanced", detector="truth")
    session.settings = ps.to_dict()
    db.commit()
    try:
        job_id = jobs.submit(session.id, ps.to_dict())
    except QueueFull as exc:
        raise HTTPException(status_code=429, detail=str(exc)) from exc
    from app.db.models import AnalysisJob

    db.refresh(session)
    return {"session": session_out(session), "job": job_out(db.get(AnalysisJob, job_id))}
