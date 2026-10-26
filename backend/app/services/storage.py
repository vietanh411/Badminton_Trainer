"""File storage with path-safety guarantees.

* Uploads are streamed to a temp file with a hard size cap; nothing is held in memory.
* Stored files are named by server-generated ids; the client's filename is only kept,
  sanitised, as a display label and never used to build a path.
* Every path handed out by the API is resolved and checked to live inside ``data_dir``.
"""

from __future__ import annotations

import hashlib
import os
import re
import shutil
import uuid
from pathlib import Path

from fastapi import UploadFile

from app.core.config import Settings

CHUNK = 1024 * 1024


class StorageError(ValueError):
    pass


class UploadTooLarge(StorageError):
    pass


def sanitize_filename(name: str | None, max_len: int = 120) -> str:
    """Display-safe version of a user-supplied filename (never used for filesystem paths)."""
    base = os.path.basename((name or "video").replace("\\", "/"))
    base = re.sub(r"[^\w.\- ()]+", "_", base, flags=re.UNICODE).strip(" .") or "video"
    return base[:max_len]


def safe_path(settings: Settings, relative: str | Path) -> Path:
    root = settings.data_dir.resolve()
    p = (root / Path(relative)).resolve()
    if p != root and root not in p.parents:
        raise StorageError("path escapes the data directory")
    return p


def relative_to_data(settings: Settings, path: Path) -> str:
    return path.resolve().relative_to(settings.data_dir.resolve()).as_posix()


async def save_upload(upload: UploadFile, settings: Settings) -> tuple[Path, str, int]:
    """Stream an upload to a temp file. Returns (temp path, sha256, size)."""
    ext = Path(upload.filename or "").suffix.lower()
    if ext not in settings.allowed_extensions:
        raise StorageError(
            f"unsupported file type '{ext or 'none'}'; allowed: {', '.join(settings.allowed_extensions)}"
        )
    settings.tmp_dir.mkdir(parents=True, exist_ok=True)
    tmp = settings.tmp_dir / f"upload_{uuid.uuid4().hex}{ext}"
    limit = settings.max_upload_mb * 1024 * 1024
    h = hashlib.sha256()
    size = 0
    try:
        with open(tmp, "wb") as fh:
            while True:
                chunk = await upload.read(CHUNK)
                if not chunk:
                    break
                size += len(chunk)
                if size > limit:
                    raise UploadTooLarge(f"file exceeds the {settings.max_upload_mb} MB upload limit")
                h.update(chunk)
                fh.write(chunk)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    if size == 0:
        tmp.unlink(missing_ok=True)
        raise StorageError("the uploaded file is empty")
    return tmp, h.hexdigest(), size


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(CHUNK), b""):
            h.update(chunk)
    return h.hexdigest()


def remove_tree(path: Path, settings: Settings) -> None:
    """Delete a directory inside data_dir (refuses anything outside it)."""
    p = safe_path(settings, relative_to_data(settings, path)) if path.is_absolute() else safe_path(settings, path)
    if p == settings.data_dir.resolve():
        raise StorageError("refusing to delete the data directory itself")
    if p.is_dir():
        shutil.rmtree(p, ignore_errors=True)
    elif p.exists():
        p.unlink()
