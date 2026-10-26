"""Automatic deletion of old sessions (BV_RETENTION_DAYS > 0)."""

from __future__ import annotations

import logging
from datetime import UTC, datetime, timedelta

from sqlalchemy import select

from app.core.config import Settings
from app.db.models import AnalysisJob, Video, VideoSession
from app.db.session import session_factory
from app.services.storage import remove_tree
from app.workers.jobs import ACTIVE

log = logging.getLogger("badmintonvision.retention")


def purge_expired(settings: Settings, now: datetime | None = None) -> int:
    """Delete sessions not updated for ``retention_days`` (with their files and orphaned videos)."""
    if settings.retention_days <= 0:
        return 0
    cutoff = (now or datetime.now(UTC)) - timedelta(days=settings.retention_days)
    removed = 0
    with session_factory()() as db:
        for s in db.scalars(select(VideoSession)).all():
            updated = s.updated_at if s.updated_at.tzinfo else s.updated_at.replace(tzinfo=UTC)
            if updated >= cutoff:
                continue
            busy = db.scalars(
                select(AnalysisJob).where(AnalysisJob.session_id == s.id, AnalysisJob.status.in_(ACTIVE))
            ).first()
            if busy:
                continue
            video_id = s.video_id
            remove_tree(settings.sessions_dir / s.id, settings)
            db.delete(s)
            db.flush()
            if db.scalars(select(VideoSession).where(VideoSession.video_id == video_id)).first() is None:
                remove_tree(settings.uploads_dir / video_id, settings)
                v = db.get(Video, video_id)
                if v is not None:
                    db.delete(v)
            removed += 1
        db.commit()
    if removed:
        log.info("retention: deleted %d session(s) older than %d days", removed, settings.retention_days)
    return removed
