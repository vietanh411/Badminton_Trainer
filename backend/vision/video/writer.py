"""Video writing helpers that produce files a browser can play where possible."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from vision.video.reader import Frame, iter_frames

# Tried in order. VP8/WebM works in the cross-platform opencv-python FFmpeg build; avc1 uses the
# OS encoder (Media Foundation on Windows) when present; mp4v always works but most browsers
# will not play it.
_CANDIDATES = [(".webm", "VP80"), (".mp4", "avc1"), (".mp4", "mp4v")]


def open_writer(stem: str | Path, fps: float, size: tuple[int, int]) -> tuple[cv2.VideoWriter, Path, str]:
    """Open the first working encoder. ``stem`` is a path without extension."""
    stem = Path(stem)
    stem.parent.mkdir(parents=True, exist_ok=True)
    for ext, fourcc in _CANDIDATES:
        path = stem.with_suffix(ext)
        writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*fourcc), fps, size)
        if writer.isOpened():
            # Probe with one frame: some backends report isOpened() but fail on write.
            return writer, path, fourcc.lower()
        writer.release()
        path.unlink(missing_ok=True)
    raise RuntimeError("no usable video encoder found in this OpenCV build")


def transcode_for_browser(src: str | Path, stem: str | Path, max_width: int = 1280, cancel=None) -> tuple[Path, str]:
    """Re-encode a video (e.g. HEVC from a phone) into a browser-playable proxy. Audio is dropped."""
    writer = None
    path = None
    codec = ""
    fps = 30.0
    cap = cv2.VideoCapture(str(src))
    if cap.isOpened():
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    cap.release()
    try:
        frame: Frame
        for frame in iter_frames(src, max_width=max_width, cancel=cancel):
            if writer is None:
                h, w = frame.image.shape[:2]
                writer, path, codec = open_writer(stem, fps, (w, h))
            writer.write(np.ascontiguousarray(frame.image))
    finally:
        if writer is not None:
            writer.release()
    if path is None:
        raise RuntimeError("source video had no frames")
    return path, codec
