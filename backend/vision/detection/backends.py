"""Concrete person detectors.

* :class:`YoloDetector` — Ultralytics YOLO (AGPL-3.0, optional dependency). Best accuracy.
* :class:`MediaPipeDetector` — MediaPipe EfficientDet-Lite (Apache-2.0). Lighter licence, lower accuracy.
* :class:`TruthDetector` — reads ground truth of synthetic scenes. Only used for the labelled
  synthetic demo and tests; it never runs on uploaded footage.
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from vision.detection.base import Detection, auto_device
from vision.video.reader import Frame


class YoloDetector:
    name = "yolo"
    license = "AGPL-3.0 (Ultralytics)"

    def __init__(self, weights: str | Path, conf: float = 0.10, imgsz: int = 640, device: str | None = None):
        try:
            from ultralytics import YOLO
        except ImportError as exc:  # pragma: no cover - depends on optional extra
            raise RuntimeError(
                "ultralytics is not installed; install the 'yolo' extra or use the mediapipe detector"
            ) from exc
        self.model = YOLO(str(weights))
        self.conf = conf
        self.imgsz = imgsz
        self.device = device or auto_device()

    def detect(self, frame: Frame) -> list[Detection]:
        res = self.model.predict(
            frame.image, classes=[0], conf=self.conf, imgsz=self.imgsz, device=self.device, verbose=False
        )[0]
        boxes = res.boxes
        if boxes is None or len(boxes) == 0:
            return []
        xyxy = boxes.xyxy.cpu().numpy()
        conf = boxes.conf.cpu().numpy()
        return [Detection(b.astype(float), float(c)) for b, c in zip(xyxy, conf, strict=True)]

    def close(self) -> None:
        self.model = None


class MediaPipeDetector:
    name = "mediapipe"
    license = "Apache-2.0 (MediaPipe EfficientDet-Lite)"

    def __init__(self, model_path: str | Path, score_threshold: float = 0.10):
        from mediapipe.tasks.python import BaseOptions, vision

        opts = vision.ObjectDetectorOptions(
            base_options=BaseOptions(model_asset_path=str(model_path)),
            running_mode=vision.RunningMode.IMAGE,
            score_threshold=score_threshold,
            category_allowlist=["person"],
            max_results=20,
        )
        self.detector = vision.ObjectDetector.create_from_options(opts)

    def detect(self, frame: Frame) -> list[Detection]:
        import mediapipe as mp

        rgb = cv2.cvtColor(frame.image, cv2.COLOR_BGR2RGB)
        result = self.detector.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb)))
        out = []
        for d in result.detections:
            bb = d.bounding_box
            score = d.categories[0].score if d.categories else 0.0
            out.append(
                Detection(
                    np.array([bb.origin_x, bb.origin_y, bb.origin_x + bb.width, bb.origin_y + bb.height], float),
                    float(score),
                )
            )
        return out

    def close(self) -> None:
        self.detector.close()


class TruthDetector:
    """Detections derived from a synthetic scene's ground truth, with configurable noise.

    ``miss_rate`` drops detections at random and ``occlusion_windows`` (list of (t0, t1, player))
    removes a player entirely, so trackers and gap handling are exercised realistically.
    """

    name = "synthetic-truth"
    license = "n/a (synthetic ground truth)"

    def __init__(
        self,
        truth: dict | str | Path,
        *,
        noise_px: float = 2.0,
        miss_rate: float = 0.03,
        occlusion_windows: list[tuple[float, float, int]] | None = None,
        seed: int = 0,
    ):
        if not isinstance(truth, dict):
            truth = json.loads(Path(truth).read_text())
        self.truth = truth
        self.times = np.array([f["t"] for f in truth["frames"]])
        self.noise = noise_px
        self.miss_rate = miss_rate
        # Scenes may declare occlusion windows in their truth file (used by the synthetic benchmark).
        self.occlusions = (
            occlusion_windows if occlusion_windows is not None else [tuple(o) for o in truth.get("occlusions", [])]
        )
        self.rng = np.random.default_rng(seed)

    def _frame_truth(self, t: float) -> dict:
        i = int(np.clip(np.searchsorted(self.times, t - 1e-4), 0, len(self.times) - 1))
        return self.truth["frames"][i]

    def detect(self, frame: Frame) -> list[Detection]:
        ft = self._frame_truth(frame.timestamp_s)
        out = []
        for p in ft["players"]:
            if any(t0 <= frame.timestamp_s <= t1 and pid == p["player"] for t0, t1, pid in self.occlusions):
                continue
            if self.rng.random() < self.miss_rate:
                continue
            b = np.asarray(p["bbox"], float) * frame.scale + self.rng.normal(0, self.noise, 4)
            out.append(Detection(b, float(np.clip(0.85 + self.rng.normal(0, 0.05), 0.3, 0.99))))
        return out

    def close(self) -> None:
        pass
