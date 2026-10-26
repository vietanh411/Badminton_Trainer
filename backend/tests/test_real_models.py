"""Pipeline with the real detector and pose models on footage of real people.

The clip is built from ``zidane.jpg``, a sample photo shipped with the Ultralytics package
(two people), panned over time to create motion. There is no court in it, so this also checks
that the pipeline degrades honestly without a calibration. Skipped when weights are missing.
"""

from __future__ import annotations

import os
from pathlib import Path

import cv2
import numpy as np
import pytest

from tests.conftest import MODELS, models_available
from vision.pipeline import PipelineInputs, PipelineSettings, run_pipeline

ultralytics = pytest.importorskip("ultralytics")
pytestmark = [pytest.mark.models, pytest.mark.slow]


@pytest.fixture(scope="module")
def people_clip(tmp_path_factory) -> Path:
    img = cv2.imread(os.path.join(os.path.dirname(ultralytics.__file__), "assets", "zidane.jpg"))
    h, w = img.shape[:2]
    out_w, out_h = 960, 540
    d = tmp_path_factory.mktemp("people")
    path = d / "people.mp4"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 15, (out_w, out_h))
    for k in range(45):  # 3 s
        x0 = int(k * 1.4)
        crop = img[:, x0 : x0 + int(w * 0.95)]
        writer.write(cv2.resize(crop, (out_w, out_h)))
    writer.release()
    return path


@pytest.mark.skipif(
    not models_available("yolo11n.pt", "pose_landmarker_lite.task"), reason="model weights not downloaded"
)
def test_yolo_bytetrack_pose_on_real_people(people_clip, tmp_path):
    from vision.detection.backends import YoloDetector
    from vision.video.reader import iter_frames

    first = next(iter_frames(people_clip))
    dets = sorted(YoloDetector(MODELS / "yolo11n.pt").detect(first), key=lambda d: -d.score)
    assert len([d for d in dets if d.score > 0.5]) >= 2
    target = dets[0]
    settings = PipelineSettings(performance_mode="fast", detector="yolo")
    inp = PipelineInputs(
        video_path=people_clip,
        video_hash="people-test",
        out_dir=tmp_path / "out",
        cache_dir=tmp_path / "c",
        models_dir=MODELS,
        settings=settings,
        selection={"t": 0.0, "bbox": target.bbox.tolist()},
    )
    summary = run_pipeline(inp)
    q = summary["quality"]
    assert q["target_coverage_pct"] > 80
    assert q["people_tracked"] <= 4  # two people; brief fragments allowed, no ID explosion
    assert summary["metrics"]["calibrated"] is False  # no court in the footage: no fake metrics
    assert summary["metrics"]["pose"]["pose_coverage_pct"] > 50
    pose = np.load(tmp_path / "out" / "pose.npz")
    xy = pose["xy"]
    ok = np.isfinite(xy).all(axis=(1, 2))
    # Landmarks are restored to full-frame coordinates (inside the image, not the crop).
    assert xy[ok][..., 0].max() < 960 and xy[ok][..., 0].min() > -50


@pytest.mark.skipif(
    not models_available("efficientdet_lite0.tflite", "pose_landmarker_lite.task"), reason="weights missing"
)
def test_mediapipe_detector_path(people_clip, tmp_path):
    settings = PipelineSettings(performance_mode="fast", detector="mediapipe")
    inp = PipelineInputs(
        video_path=people_clip,
        video_hash="people-test-mp",
        out_dir=tmp_path / "out",
        cache_dir=tmp_path / "c",
        models_dir=MODELS,
        settings=settings,
        selection=None,
    )
    summary = run_pipeline(inp)
    assert summary["analysis"]["detector"] == "mediapipe"
    assert summary["quality"]["selection"]["auto"] is True
    assert summary["quality"]["target_coverage_pct"] > 50
