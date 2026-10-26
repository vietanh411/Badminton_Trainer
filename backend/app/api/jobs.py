"""Job status (polling and Server-Sent Events) and cancellation."""

from __future__ import annotations

import asyncio
import json

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session

from app.core.deps import get_jobs
from app.db.models import AnalysisJob
from app.db.session import get_db, session_factory
from app.services.serialize import job_out
from app.workers.jobs import TERMINAL, JobManager

router = APIRouter(prefix="/api/jobs", tags=["jobs"])


@router.get("/{job_id}", summary="Job status and progress")
def get_job(job_id: str, db: Session = Depends(get_db)):
    job = db.get(AnalysisJob, job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="job not found")
    return job_out(job)


@router.post("/{job_id}/cancel", summary="Request cancellation (takes effect at the next frame)")
def cancel_job(job_id: str, db: Session = Depends(get_db), jobs: JobManager = Depends(get_jobs)):
    job = db.get(AnalysisJob, job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="job not found")
    if not jobs.cancel(job_id):
        raise HTTPException(status_code=409, detail=f"job already {job.status.lower()}")
    db.refresh(job)
    return job_out(job)


@router.get("/{job_id}/events", summary="Server-Sent Events stream of job progress")
async def job_events(job_id: str, request: Request):
    def snapshot() -> dict | None:
        with session_factory()() as db:
            job = db.get(AnalysisJob, job_id)
            return json.loads(json.dumps(job_out(job), default=str)) if job else None

    if snapshot() is None:
        raise HTTPException(status_code=404, detail="job not found")

    async def stream():
        last = None
        while True:
            if await request.is_disconnected():
                return
            snap = await asyncio.to_thread(snapshot)
            if snap is None:
                return
            if snap != last:
                yield f"data: {json.dumps(snap)}\n\n"
                last = snap
            if snap["status"] in TERMINAL:
                return
            await asyncio.sleep(0.5)

    return StreamingResponse(
        stream(), media_type="text/event-stream", headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}
    )
