"""FastAPI dependencies: settings, DB session, API-key guard, job manager."""

from __future__ import annotations

import hmac

from fastapi import Depends, Header, HTTPException, Request
from sqlalchemy.orm import Session

from app.core.config import Settings, get_settings
from app.db.models import VideoSession
from app.db.session import get_db
from app.workers.jobs import JobManager


def require_api_key(
    x_api_key: str | None = Header(default=None), settings: Settings = Depends(get_settings), request: Request = None
) -> None:
    """When BV_API_KEY is set, every API call must present it (header X-API-Key, or ?key= for media URLs)."""
    if not settings.api_key:
        return
    supplied = x_api_key or (request.query_params.get("key") if request is not None else None)
    if not supplied or not hmac.compare_digest(supplied, settings.api_key):
        raise HTTPException(status_code=401, detail="missing or invalid API key")


def get_jobs(request: Request) -> JobManager:
    return request.app.state.jobs


def get_session_or_404(session_id: str, db: Session = Depends(get_db)) -> VideoSession:
    s = db.get(VideoSession, session_id)
    if s is None:
        raise HTTPException(status_code=404, detail="session not found")
    return s
