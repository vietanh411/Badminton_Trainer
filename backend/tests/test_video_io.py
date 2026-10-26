import numpy as np
import pytest

from vision.video.reader import VideoError, iter_frames, make_thumbnail, probe_video, read_frame_at, sniff_container


def test_probe_metadata(synthetic_clip):
    meta = probe_video(synthetic_clip["video"])
    assert (meta.width, meta.height) == (640, 360)
    assert meta.fps == pytest.approx(30.0, abs=0.01)
    assert meta.duration_s == pytest.approx(8.0, abs=0.1)
    assert meta.container in ("webm", "mp4")
    assert meta.file_size > 0


def test_sampling_is_uniform_and_timestamped(synthetic_clip):
    frames = list(iter_frames(synthetic_clip["video"], target_fps=10, max_width=320))
    ts = np.array([f.timestamp_s for f in frames])
    assert len(frames) == pytest.approx(80, abs=2)
    assert np.allclose(np.diff(ts), 0.1, atol=0.004)
    assert frames[0].image.shape[1] == 320 and frames[0].scale == pytest.approx(0.5)
    assert [f.sample_index for f in frames] == list(range(len(frames)))


def test_full_rate_and_window(synthetic_clip):
    frames = list(iter_frames(synthetic_clip["video"], start_s=2.0, end_s=3.0))
    assert 29 <= len(frames) <= 32
    assert frames[0].timestamp_s == pytest.approx(2.0, abs=0.05)


def test_cancellation_stops_iteration(synthetic_clip):
    import threading

    ev = threading.Event()
    n = 0
    for _ in iter_frames(synthetic_clip["video"], cancel=ev):
        n += 1
        if n == 5:
            ev.set()
    assert n == 5


def test_read_frame_and_thumbnail(synthetic_clip, tmp_path):
    img, t = read_frame_at(synthetic_clip["video"], 2.0)
    assert img.shape == (360, 640, 3)
    assert t == pytest.approx(2.0, abs=0.1)
    thumb = make_thumbnail(synthetic_clip["video"], tmp_path / "t.jpg", width=200)
    assert thumb.exists() and thumb.stat().st_size > 1000


def test_rejects_non_video(tmp_path):
    p = tmp_path / "notes.mp4"
    p.write_text("this is not a video")
    with pytest.raises(VideoError, match="not an MP4"):
        probe_video(p)


def test_rejects_corrupted_mp4(tmp_path):
    p = tmp_path / "broken.mp4"
    p.write_bytes(
        b"\x00\x00\x00\x18ftypisom" + bytes(np.random.default_rng(0).integers(0, 255, 50_000, dtype=np.uint8))
    )
    with pytest.raises(VideoError):
        probe_video(p)


def test_rejects_truncated_video(synthetic_clip, tmp_path):
    data = synthetic_clip["video"].read_bytes()
    p = tmp_path / f"trunc{synthetic_clip['video'].suffix}"
    p.write_bytes(data[:200])
    with pytest.raises(VideoError):
        probe_video(p)


def test_container_sniffing(tmp_path, synthetic_clip):
    assert sniff_container(synthetic_clip["video"]) in ("webm", "mp4")
    p = tmp_path / "x.avi"
    p.write_bytes(b"RIFF\x00\x00\x00\x00AVI LIST")
    assert sniff_container(p) == "avi"


def test_missing_file(tmp_path):
    with pytest.raises(VideoError, match="not found"):
        probe_video(tmp_path / "nope.mp4")
