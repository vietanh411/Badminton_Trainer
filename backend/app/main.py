"""FastAPI application factory."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import Depends, FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from app.api import benchmark, demo, jobs, sessions, system, videos
from app.core.config import Settings, get_settings
from app.core.deps import require_api_key
from app.db.session import init_engine, run_migrations
from app.workers.jobs import JobManager
from vision.pipeline import PIPELINE_VERSION
from vision.video.reader import VideoError

log = logging.getLogger("badmintonvision")


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or get_settings()
    logging.basicConfig(level=settings.log_level, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        settings.ensure_dirs()
        init_engine(settings)
        run_migrations(settings)
        manager = JobManager(settings)
        n = manager.recover_interrupted()
        if n:
            log.warning("marked %d interrupted job(s) as failed", n)
        app.state.jobs = manager
        yield
        manager.shutdown()

    app = FastAPI(
        title="BadmintonVision AI",
        version=PIPELINE_VERSION,
        description="Badminton video analytics: court calibration, player tracking, pose, court-movement heatmaps, "
        "movement metrics and experimental stroke analysis.",
        lifespan=lifespan,
    )
    app.dependency_overrides[get_settings] = lambda: settings
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
        expose_headers=["X-Frame-Time", "X-Image-Width", "X-Image-Height", "Content-Disposition"],
    )

    guard = [Depends(require_api_key)]
    app.include_router(system.public_router)
    for r in (
        system.router,
        videos.router,
        sessions.router,
        sessions.stroke_router,
        jobs.router,
        demo.router,
        benchmark.router,
    ):
        app.include_router(r, dependencies=guard)

    @app.exception_handler(VideoError)
    async def video_error(_request: Request, exc: VideoError):
        return JSONResponse({"detail": str(exc)}, status_code=422)

    # Serve the built frontend (frontend/dist) when present, so one process can host everything.
    dist = Path(__file__).resolve().parents[2] / "frontend" / "dist"
    if dist.exists():
        app.mount("/assets", StaticFiles(directory=dist / "assets"), name="assets")

        @app.get("/{full_path:path}", include_in_schema=False)
        async def spa(full_path: str):
            if full_path.startswith("api/"):
                return JSONResponse({"detail": "not found"}, status_code=404)
            candidate = (dist / full_path).resolve()
            if full_path and dist in candidate.parents and candidate.is_file():
                return FileResponse(candidate)
            return FileResponse(dist / "index.html")

    return app


app = create_app()
