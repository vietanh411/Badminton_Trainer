"""Streaming, timestamp-aware video decoding with OpenCV.

Frames are decoded one at a time and never accumulated, so memory use is independent of the
video length. Rotation metadata (phone footage) is applied by OpenCV's FFmpeg backend when
``CAP_PROP_ORIENTATION_AUTO`` is on, which is the default; we report the rotation we saw.
"""

from __future__ import annotations

import threading
from collections.abc import Iterator
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np

# Codecs that mainstream browsers can play inside <video> without transcoding.
BROWSER_CODECS = {"h264", "avc1", "avc3", "vp80", "vp08", "vp90", "vp09", "av01"}

# Container signatures checked against the first bytes of the upload (MIME types sent by the
# client are not trusted).
_SIGNATURES = [
    ("mp4", lambda b: len(b) >= 12 and b[4:8] == b"ftyp"),
    ("webm", lambda b: b[:4] == b"\x1a\x45\xdf\xa3"),
    ("avi", lambda b: b[:4] == b"RIFF" and b[8:12] == b"AVI "),
]


class VideoError(ValueError):
    """The file is not a decodable video or violates configured limits."""


@dataclass
class VideoMetadata:
    width: int
    height: int
    fps: float
    frame_count: int
    duration_s: float
    codec: str
    rotation: int
    container: str
    file_size: int
    browser_playable: bool

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Frame:
    index: int  # index in the source video
    sample_index: int  # index among the sampled frames
    timestamp_s: float
    image: np.ndarray  # BGR, possibly resized
    scale: float  # resized / original


def sniff_container(path: str | Path) -> str | None:
    with open(path, "rb") as fh:
        head = fh.read(16)
    for name, test in _SIGNATURES:
        if test(head):
            return name
    return None


def _fourcc_to_str(value: float) -> str:
    code = int(value)
    if code <= 0:
        return "unknown"
    try:
        s = code.to_bytes(4, "little").decode("ascii", errors="replace").strip("\x00 ").lower()
    except OverflowError:
        return "unknown"
    return s or "unknown"


def probe_video(path: str | Path) -> VideoMetadata:
    """Read container-level metadata and verify that at least one frame decodes."""
    path = Path(path)
    if not path.is_file():
        raise VideoError("video file not found")
    container = sniff_container(path)
    if container is None:
        raise VideoError("unsupported or corrupted file: not an MP4/MOV, WebM/MKV or AVI container")
    cap = cv2.VideoCapture(str(path))
    try:
        if not cap.isOpened():
            raise VideoError("the video could not be opened (corrupted file or unsupported codec)")
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        codec = _fourcc_to_str(cap.get(cv2.CAP_PROP_FOURCC))
        rotation = int(cap.get(cv2.CAP_PROP_ORIENTATION_META) or 0) if hasattr(cv2, "CAP_PROP_ORIENTATION_META") else 0
        ok, frame = cap.read()
        if not ok or frame is None:
            raise VideoError("no decodable frames (corrupted file or unsupported codec)")
        # After auto-orientation the decoded frame size is authoritative.
        height, width = frame.shape[:2]
        if fps <= 0 or fps > 1000:
            raise VideoError("could not determine a valid frame rate")
        if count <= 0:
            count = _count_frames(cap) + 1
        duration = count / fps
    finally:
        cap.release()
    if width < 64 or height < 64:
        raise VideoError(f"resolution {width}x{height} is too small to analyse")
    return VideoMetadata(
        width=width,
        height=height,
        fps=round(fps, 3),
        frame_count=count,
        duration_s=round(duration, 3),
        codec=codec,
        rotation=rotation,
        container=container,
        file_size=path.stat().st_size,
        browser_playable=codec in BROWSER_CODECS,
    )


def _count_frames(cap: cv2.VideoCapture) -> int:
    n = 0
    while cap.grab():
        n += 1
    return n


def resize_to_width(image: np.ndarray, max_width: int | None) -> tuple[np.ndarray, float]:
    h, w = image.shape[:2]
    if not max_width or w <= max_width:
        return image, 1.0
    scale = max_width / w
    return cv2.resize(image, (max_width, round(h * scale)), interpolation=cv2.INTER_AREA), scale


def sample_step(source_fps: float, target_fps: float | None) -> int:
    """Keep every n-th source frame so samples are evenly spaced (25 fps → 15 fps target gives 12.5 fps).

    Picking frames by timestamp instead would alternate 1- and 2-frame gaps, and every smoother and
    derivative downstream prefers uniform spacing.
    """
    if not target_fps or target_fps >= source_fps:
        return 1
    return max(1, int(source_fps / target_fps + 0.5))


def iter_frames(
    path: str | Path,
    *,
    target_fps: float | None = None,
    max_width: int | None = None,
    start_s: float = 0.0,
    end_s: float | None = None,
    cancel: threading.Event | None = None,
) -> Iterator[Frame]:
    """Yield every n-th frame (n = ``sample_step``, about ``target_fps``) with real presentation timestamps.

    Skipped frames are only ``grab``-bed (demuxed and decoded but not converted to BGR), which
    is noticeably cheaper than ``read``. Timestamps come from the decoder (``CAP_PROP_POS_MSEC``)
    so variable-frame-rate phone footage is handled; when the backend reports nothing useful we
    fall back to ``index / fps``.
    """
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise VideoError("the video could not be opened")
    try:
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
        if start_s > 0:
            cap.set(cv2.CAP_PROP_POS_MSEC, start_s * 1000.0)
        index = int(round(start_s * fps)) - 1
        step = sample_step(fps, target_fps)
        decoded = -1
        last_t = -1.0
        sample_index = 0
        while True:
            if cancel is not None and cancel.is_set():
                return
            if not cap.grab():
                return
            index += 1
            decoded += 1
            pos_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
            t = pos_ms / 1000.0 if pos_ms and pos_ms > 0 else index / fps
            if t <= last_t:  # non-monotonic or missing timestamps
                t = max(index / fps, last_t + 1e-3)
            last_t = t
            if end_s is not None and t > end_s:
                return
            if decoded % step:
                continue
            ok, frame = cap.retrieve()
            if not ok or frame is None:
                continue
            image, scale = resize_to_width(frame, max_width)
            yield Frame(index=index, sample_index=sample_index, timestamp_s=t, image=image, scale=scale)
            sample_index += 1
    finally:
        cap.release()


def read_frame_at(path: str | Path, t: float, max_width: int | None = None) -> tuple[np.ndarray, float]:
    """Decode the frame closest to time ``t`` (seconds). Returns (image, actual_timestamp)."""
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise VideoError("the video could not be opened")
    try:
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
        cap.set(cv2.CAP_PROP_POS_MSEC, max(0.0, t) * 1000.0)
        ok, frame = cap.read()
        if not ok or frame is None:
            # Seeking past the end on some containers fails; step back one frame.
            cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) - 1))
            ok, frame = cap.read()
            if not ok or frame is None:
                raise VideoError("could not decode a frame at that time")
        actual = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0 - 1.0 / fps
        image, _ = resize_to_width(frame, max_width)
        return image, max(0.0, actual)
    finally:
        cap.release()


def make_thumbnail(path: str | Path, out_path: str | Path, at_s: float = 1.0, width: int = 480) -> Path:
    meta_t = at_s
    try:
        image, _ = read_frame_at(path, meta_t, max_width=width)
    except VideoError:
        image, _ = read_frame_at(path, 0.0, max_width=width)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), image, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return out_path
