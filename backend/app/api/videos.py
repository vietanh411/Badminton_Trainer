"""Video upload, metadata, preview streaming and frame extraction."""

from __future__ import annotations

import asyncio
import shutil
from pathlib import Path

import cv2
from fastapi import APIRouter, Depends, File, HTTPException, Query, Response, UploadFile
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session

from app.core.config import Settings, get_settings
from app.db.models import Video, VideoSession
from app.db.session import get_db
from app.services.serialize import session_out, video_out
from app.services.storage import (
    StorageError,
    UploadTooLarge,
    relative_to_data,
    safe_path,
    sanitize_filename,
    save_upload,
)
from vision.video.reader import VideoError, make_thumbnail, probe_video, read_frame_at

router = APIRouter(prefix="/api/videos", tags=["videos"])

_MEDIA_TYPES = {
    ".mp4": "video/mp4",
    ".m4v": "video/mp4",
    ".mov": "video/quicktime",
    ".webm": "video/webm",
    ".mkv": "video/x-matroska",
    ".avi": "video/x-msvideo",
}


def register_video(
    db: Session,
    settings: Settings,
    tmp: Path,
    sha: str,
    size: int,
    display_name: str,
    *,
    is_synthetic: bool = False,
    truth: Path | None = None,
    session_name: str | None = None,
    is_demo: bool = False,
) -> tuple[Video, VideoSession]:
    """Validate a file already on disk, move it into storage and create its default session."""
    try:
        meta = probe_video(tmp)
    except VideoError:
        tmp.unlink(missing_ok=True)
        raise
    if meta.duration_s > settings.max_duration_s:
        tmp.unlink(missing_ok=True)
        raise VideoError(f"video is {meta.duration_s:.0f} s long; the limit is {settings.max_duration_s:.0f} s")
    video = Video(
        original_filename=display_name,
        stored_path="",
        sha256=sha,
        size_bytes=size,
        container=meta.container,
        codec=meta.codec,
        width=meta.width,
        height=meta.height,
        fps=meta.fps,
        frame_count=meta.frame_count,
        duration_s=meta.duration_s,
        rotation=meta.rotation,
        browser_playable=meta.browser_playable,
        is_synthetic=is_synthetic,
    )
    db.add(video)
    db.flush()
    vdir = settings.uploads_dir / video.id
    vdir.mkdir(parents=True, exist_ok=True)
    dest = vdir / f"source{tmp.suffix.lower()}"
    shutil.move(str(tmp), dest)
    video.stored_path = relative_to_data(settings, dest)
    if truth is not None:
        tdest = vdir / "truth.json"
        shutil.move(str(truth), tdest)
        video.truth_path = relative_to_data(settings, tdest)
    try:
        thumb = make_thumbnail(dest, vdir / "thumbnail.jpg", at_s=min(1.0, meta.duration_s / 2))
        video.thumbnail_path = relative_to_data(settings, thumb)
    except VideoError:
        pass
    session = VideoSession(
        video_id=video.id,
        name=session_name or Path(display_name).stem[:200] or "Session",
        data_source="synthetic" if is_synthetic else "upload",
        is_demo=is_demo,
        settings={},
        corrections=[],
    )
    db.add(session)
    db.commit()
    db.refresh(session)
    return video, session


@router.post("", status_code=201, summary="Upload a video (multipart). Creates a video and a draft session.")
async def upload_video(
    file: UploadFile = File(...), db: Session = Depends(get_db), settings: Settings = Depends(get_settings)
):
    try:
        tmp, sha, size = await save_upload(file, settings)
    except UploadTooLarge as exc:
        raise HTTPException(status_code=413, detail=str(exc)) from exc
    except StorageError as exc:
        raise HTTPException(status_code=415, detail=str(exc)) from exc
    display = sanitize_filename(file.filename)
    try:
        # Probing decodes a frame; keep it off the event loop.
        video, session = await asyncio.to_thread(register_video, db, settings, tmp, sha, size, display)
    except VideoError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return {"video": video_out(video), "session": session_out(session)}


def _video_or_404(db: Session, video_id: str) -> Video:
    v = db.get(Video, video_id)
    if v is None:
        raise HTTPException(status_code=404, detail="video not found")
    return v


@router.get("/{video_id}", summary="Video metadata")
def get_video(video_id: str, db: Session = Depends(get_db)):
    return video_out(_video_or_404(db, video_id))


@router.get("/{video_id}/preview", summary="Stream the video (HTTP range requests supported)")
def preview(video_id: str, db: Session = Depends(get_db), settings: Settings = Depends(get_settings)):
    v = _video_or_404(db, video_id)
    rel = v.proxy_path if (v.proxy_path and not v.browser_playable) else v.stored_path
    path = safe_path(settings, rel)
    if not path.exists():
        raise HTTPException(status_code=404, detail="video file missing")
    return FileResponse(
        path,
        media_type=_MEDIA_TYPES.get(path.suffix.lower(), "application/octet-stream"),
        headers={"Cache-Control": "private, max-age=3600"},
    )


@router.get("/{video_id}/thumbnail", summary="JPEG thumbnail")
def thumbnail(video_id: str, db: Session = Depends(get_db), settings: Settings = Depends(get_settings)):
    v = _video_or_404(db, video_id)
    if not v.thumbnail_path:
        raise HTTPException(status_code=404, detail="no thumbnail")
    return FileResponse(safe_path(settings, v.thumbnail_path), media_type="image/jpeg")


@router.get("/{video_id}/frame", summary="Decode one frame as JPEG (used by calibration and player selection)")
async def frame(
    video_id: str,
    t: float = Query(0.0, ge=0),
    max_width: int = Query(1920, ge=160, le=3840),
    db: Session = Depends(get_db),
    settings: Settings = Depends(get_settings),
):
    v = _video_or_404(db, video_id)
    path = safe_path(settings, v.stored_path)
    try:
        img, actual = await asyncio.to_thread(read_frame_at, path, min(t, max(0.0, v.duration_s - 0.05)), max_width)
    except VideoError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 90])
    if not ok:
        raise HTTPException(status_code=500, detail="could not encode frame")
    return Response(
        buf.tobytes(),
        media_type="image/jpeg",
        headers={
            "X-Frame-Time": f"{actual:.3f}",
            "X-Image-Width": str(img.shape[1]),
            "X-Image-Height": str(img.shape[0]),
            "Cache-Control": "private, max-age=600",
        },
    )
