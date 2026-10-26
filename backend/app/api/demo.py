"""Clearly-labelled synthetic demo sessions.

The demo renders a simulated rally video with the synthetic scene generator and analyses it
with ground-truth "detectors". Everything about these sessions is flagged ``synthetic`` in the
database, API responses, exports and the UI. They exist so the product can be explored without
real footage; their metrics describe a simulation, not an athlete.

The request returns immediately: the simulator is deterministic, so the session, calibration and
player selection are derived from the scene description, and the video itself is rendered as the
first step of the background job (see ``app/services/demo.py``).
"""

from __future__ import annotations

import json

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from app.core.config import Settings, get_settings
from app.core.deps import get_jobs
from app.db.models import AnalysisJob, CourtCalibration, Video, VideoSession
from app.db.session import get_db
from app.services.demo import DEMO_FPS, DEMO_SIZE, SPEC_NAME, demo_scene
from app.services.serialize import job_out, session_out
from app.services.storage import relative_to_data
from app.workers.jobs import JobManager, QueueFull
from vision.calibration.homography import fit_homography
from vision.pipeline import PipelineSettings
from vision.synthetic.scene import court_keypoint_pixels

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


@router.post("", status_code=201, summary="Create a synthetic demo session and start its analysis")
def create_demo(
    variant: int = Query(1, ge=1, le=2),
    duration_s: float = Query(45.0, ge=10, le=120),
    db: Session = Depends(get_db),
    settings: Settings = Depends(get_settings),
    jobs: JobManager = Depends(get_jobs),
):
    if not settings.enable_demo:
        raise HTTPException(status_code=404, detail="demo mode is disabled")
    v = VARIANTS[variant]
    spec = {"seed": v["seed"], "intensity": v["intensity"], "duration_s": duration_s}
    scene = demo_scene(spec)  # cheap: simulation only, no rendering
    width, height = DEMO_SIZE
    video = Video(
        original_filename=f"synthetic_demo_{variant}.webm",
        stored_path="",
        # Deterministic content id: identical demos share the detection/pose cache.
        sha256=f"synthetic-demo-{v['seed']}-{v['intensity']}-{duration_s}-{width}x{height}",
        size_bytes=0,
        container="webm",
        codec="pending",
        width=width,
        height=height,
        fps=DEMO_FPS,
        frame_count=scene.n_frames,
        duration_s=duration_s,
        rotation=0,
        browser_playable=True,
        is_synthetic=True,
    )
    db.add(video)
    db.flush()
    vdir = settings.uploads_dir / video.id
    vdir.mkdir(parents=True, exist_ok=True)
    (vdir / SPEC_NAME).write_text(json.dumps(spec))
    video.stored_path = relative_to_data(settings, vdir / "source.webm")  # final name set after rendering
    session = VideoSession(
        video_id=video.id, name=v["name"], data_source="synthetic", is_demo=True, settings={}, corrections=[]
    )
    db.add(session)
    db.flush()
    # The simulator knows the exact camera, so the demo calibration uses exact landmark pixels.
    d = fit_homography(
        court_keypoint_pixels(scene.camera, CAL_POINTS),
        point_ids=CAL_POINTS,
        method="manual",
        image_size=(width, height),
    ).to_dict()
    db.add(
        CourtCalibration(
            session_id=session.id,
            method="manual",
            frame_time_s=1.0,
            image_width=width,
            image_height=height,
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
    i = int(round(scene.fps))  # target player at t = 1 s
    session.player_selection = {"t": i / scene.fps, "bbox": scene.frame_truth(i)[0]["bbox"]}
    ps = PipelineSettings(performance_mode="balanced", detector="truth")
    session.settings = ps.to_dict()
    db.commit()
    try:
        job_id = jobs.submit(session.id, ps.to_dict())
    except QueueFull as exc:
        raise HTTPException(status_code=429, detail=str(exc)) from exc
    db.refresh(session)
    return {"session": session_out(session), "job": job_out(db.get(AnalysisJob, job_id))}
