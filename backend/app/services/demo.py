"""Deferred rendering of synthetic demo videos (runs inside the analysis job)."""

from __future__ import annotations

import json
import threading

from app.core.config import Settings
from app.db.models import Video
from app.db.session import session_factory
from app.services.storage import relative_to_data
from vision.synthetic.scene import SyntheticScene, make_scene, write_scene
from vision.video.reader import make_thumbnail, probe_video

DEMO_SIZE = (1280, 720)
DEMO_FPS = 30.0
SPEC_NAME = "demo_spec.json"


def demo_scene(spec: dict) -> SyntheticScene:
    return make_scene(
        duration=spec["duration_s"],
        fps=DEMO_FPS,
        width=DEMO_SIZE[0],
        height=DEMO_SIZE[1],
        seed=spec["seed"],
        intensity=spec["intensity"],
    )


def ensure_demo_rendered(video_id: str, settings: Settings, progress, cancel: threading.Event | None = None) -> None:
    """Render the demo video and its ground truth if this synthetic video has not been rendered yet."""
    with session_factory()() as db:
        video = db.get(Video, video_id)
        if not video.is_synthetic or video.truth_path:
            return
        vdir = settings.uploads_dir / video.id
        spec_path = vdir / SPEC_NAME
        if not spec_path.exists():
            return
    spec = json.loads(spec_path.read_text())
    progress("VALIDATING", 0.1, f"rendering the {spec['duration_s']:.0f} s synthetic demo video")
    scene = demo_scene(spec)
    path, truth = write_scene(scene, vdir / "source")
    if cancel is not None and cancel.is_set():
        from vision.pipeline import Cancelled

        raise Cancelled()
    meta = probe_video(path)
    thumb = make_thumbnail(path, vdir / "thumbnail.jpg", at_s=1.0)
    truth.replace(vdir / "truth.json")
    with session_factory()() as db:
        video = db.get(Video, video_id)
        video.stored_path = relative_to_data(settings, path)
        video.truth_path = relative_to_data(settings, vdir / "truth.json")
        video.thumbnail_path = relative_to_data(settings, thumb)
        video.size_bytes = path.stat().st_size
        video.codec = meta.codec
        video.container = meta.container
        video.frame_count = meta.frame_count
        video.browser_playable = meta.browser_playable
        db.commit()
