"""Background analysis jobs.

Jobs run in a bounded thread pool, never on the API event loop. OpenCV, PyTorch and MediaPipe
release the GIL in their heavy kernels, so threads give real parallelism with the web server
while keeping cancellation simple (a per-job ``threading.Event`` checked between frames).

Each job builds its own detector/tracker/pose objects, so no model or tracker state is shared
between jobs. Progress is written to the database (throttled) so polling and SSE clients see it.
"""

from __future__ import annotations

import logging
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime

from sqlalchemy import select

from app.core.config import Settings
from app.db.models import AnalysisJob, VideoSession
from app.db.session import session_factory

log = logging.getLogger("badmintonvision.jobs")

ACTIVE = ("QUEUED", "VALIDATING", "CALIBRATING", "DETECTING", "TRACKING", "ESTIMATING_POSE", "ANALYZING", "RENDERING")
TERMINAL = ("COMPLETED", "FAILED", "CANCELLED")

# Rough share of total runtime per stage, used to turn stage progress into overall progress.
_STAGE_WEIGHTS = {
    "VALIDATING": 0.02,
    "CALIBRATING": 0.03,
    "DETECTING": 0.45,
    "TRACKING": 0.03,
    "ESTIMATING_POSE": 0.35,
    "ANALYZING": 0.06,
    "RENDERING": 0.06,
}
_ORDER = list(_STAGE_WEIGHTS)


def overall_progress(stage: str, frac: float) -> float:
    done = sum(_STAGE_WEIGHTS[s] for s in _ORDER[: _ORDER.index(stage)]) if stage in _ORDER else 0.0
    return round(min(0.999, done + _STAGE_WEIGHTS.get(stage, 0.0) * max(0.0, min(1.0, frac))), 4)


class QueueFull(RuntimeError):
    pass


class JobManager:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.executor = ThreadPoolExecutor(
            max_workers=max(1, settings.max_concurrent_jobs), thread_name_prefix="bv-job"
        )
        self.cancel_events: dict[str, threading.Event] = {}
        self.lock = threading.Lock()

    def recover_interrupted(self) -> int:
        """Mark jobs that were running when the server stopped as failed."""
        with session_factory()() as db:
            jobs = db.scalars(select(AnalysisJob).where(AnalysisJob.status.in_(ACTIVE))).all()
            for j in jobs:
                j.status = "FAILED"
                j.error = "the server restarted while this job was running; start the analysis again"
                j.finished_at = datetime.now(UTC)
                s = db.get(VideoSession, j.session_id)
                if s is not None and s.latest_job_id == j.id:
                    s.status = "failed"
            db.commit()
            return len(jobs)

    def active_count(self) -> int:
        with session_factory()() as db:
            return len(db.scalars(select(AnalysisJob.id).where(AnalysisJob.status.in_(ACTIVE))).all())

    def submit(self, session_id: str, settings_snapshot: dict) -> str:
        with self.lock:
            if self.active_count() >= self.settings.max_queued_jobs:
                raise QueueFull("too many analyses are queued; wait for one to finish")
            with session_factory()() as db:
                s = db.get(VideoSession, session_id)
                running = db.scalars(
                    select(AnalysisJob).where(AnalysisJob.session_id == session_id, AnalysisJob.status.in_(ACTIVE))
                ).first()
                if running is not None:
                    raise QueueFull("an analysis for this session is already running")
                job = AnalysisJob(
                    session_id=session_id,
                    status="QUEUED",
                    settings=settings_snapshot,
                    progress=0.0,
                    message="waiting for a free worker",
                )
                db.add(job)
                db.flush()
                s.latest_job_id = job.id
                s.status = "queued"
                db.commit()
                job_id = job.id
            self.cancel_events[job_id] = threading.Event()
        self.executor.submit(self._run, job_id)
        return job_id

    def cancel(self, job_id: str) -> bool:
        ev = self.cancel_events.get(job_id)
        with session_factory()() as db:
            job = db.get(AnalysisJob, job_id)
            if job is None or job.status in TERMINAL:
                return False
            job.cancel_requested = True
            if job.status == "QUEUED":
                job.status = "CANCELLED"
                job.finished_at = datetime.now(UTC)
                job.message = "cancelled before it started"
                s = db.get(VideoSession, job.session_id)
                if s is not None:
                    s.status = "cancelled"
            db.commit()
        if ev is not None:
            ev.set()
        return True

    def _update(self, job_id: str, **fields) -> None:
        with session_factory()() as db:
            job = db.get(AnalysisJob, job_id)
            if job is None:
                return
            for k, v in fields.items():
                setattr(job, k, v)
            db.commit()

    def _run(self, job_id: str) -> None:
        from app.services.analysis import execute_job  # local import keeps worker startup light

        cancel = self.cancel_events.setdefault(job_id, threading.Event())
        with session_factory()() as db:
            job = db.get(AnalysisJob, job_id)
            if job is None or job.status == "CANCELLED":
                return
            job.status = "VALIDATING"
            job.started_at = datetime.now(UTC)
            s = db.get(VideoSession, job.session_id)
            s.status = "processing"
            db.commit()

        last = {"t": 0.0, "stage": None}

        def progress(stage: str, frac: float, message: str) -> None:
            now = time.time()
            if stage == last["stage"] and now - last["t"] < 0.5 and frac < 1.0:
                return
            last.update(t=now, stage=stage)
            self._update(job_id, status=stage, stage=stage, progress=overall_progress(stage, frac), message=message)

        try:
            timings = execute_job(job_id, self.settings, progress, cancel)
            self._update(
                job_id,
                status="COMPLETED",
                progress=1.0,
                message="analysis complete",
                timings=timings,
                finished_at=datetime.now(UTC),
            )
            self._set_session_status(job_id, "completed")
        except Exception as exc:  # noqa: BLE001 - every failure must end the job cleanly
            from vision.pipeline import Cancelled

            if isinstance(exc, Cancelled) or cancel.is_set():
                self._update(job_id, status="CANCELLED", message="cancelled by user", finished_at=datetime.now(UTC))
                self._set_session_status(job_id, "cancelled")
            else:
                log.error("job %s failed: %s", job_id, exc)
                log.debug(traceback.format_exc())
                self._update(
                    job_id,
                    status="FAILED",
                    error=str(exc) or exc.__class__.__name__,
                    message="analysis failed",
                    finished_at=datetime.now(UTC),
                )
                self._set_session_status(job_id, "failed")
        finally:
            self.cancel_events.pop(job_id, None)

    def _set_session_status(self, job_id: str, status: str) -> None:
        with session_factory()() as db:
            job = db.get(AnalysisJob, job_id)
            s = db.get(VideoSession, job.session_id) if job else None
            if s is not None and s.latest_job_id == job_id:
                s.status = status
                db.commit()

    def shutdown(self) -> None:
        for ev in self.cancel_events.values():
            ev.set()
        self.executor.shutdown(wait=False, cancel_futures=True)
