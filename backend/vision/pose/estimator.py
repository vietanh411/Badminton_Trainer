"""Pose estimation on the selected player's crop, with coordinates restored to the full frame."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import cv2
import numpy as np

from vision.pose.landmarks import NUM_LANDMARKS
from vision.video.reader import Frame


@dataclass
class PoseResult:
    xy: np.ndarray  # (33, 2) analysis-frame pixels
    z: np.ndarray  # (33,) MediaPipe relative depth (hip-centred, roughly the scale of x); not metric
    visibility: np.ndarray  # (33,)
    presence: np.ndarray  # (33,)


class PoseEstimator(Protocol):
    name: str

    def estimate(self, frame: Frame, bbox: np.ndarray) -> PoseResult | None: ...

    def close(self) -> None: ...


def crop_box(
    bbox: np.ndarray, image_shape: tuple[int, ...], margin: float = 0.25, min_aspect: float = 0.75
) -> tuple[int, int, int, int]:
    """Expand a person box by ``margin`` and widen it so raised arms and rackets stay in view."""
    h_img, w_img = image_shape[:2]
    x1, y1, x2, y2 = [float(v) for v in bbox]
    w, h = x2 - x1, y2 - y1
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    h2 = h * (1 + 2 * margin)
    w2 = max(w * (1 + 2 * margin), h2 * min_aspect)
    X1 = int(np.clip(round(cx - w2 / 2), 0, w_img - 1))
    X2 = int(np.clip(round(cx + w2 / 2), X1 + 1, w_img))
    Y1 = int(np.clip(round(cy - h2 / 2), 0, h_img - 1))
    Y2 = int(np.clip(round(cy + h2 / 2), Y1 + 1, h_img))
    return X1, Y1, X2, Y2


class MediaPipePoseEstimator:
    """MediaPipe Pose Landmarker (Apache-2.0) run on a crop around the tracked player.

    ``running_mode="image"`` runs person detection + landmarks on every crop. It is slower than
    ``"video"`` but robust to the crop moving and to gaps in tracking, which matter here because
    the crop window follows the player and frames without a detection are skipped.
    """

    def __init__(
        self,
        model_path: str | Path,
        *,
        margin: float = 0.25,
        running_mode: str = "image",
        min_detection_confidence: float = 0.4,
        min_presence_confidence: float = 0.4,
    ):
        from mediapipe.tasks.python import BaseOptions, vision

        self._video = running_mode == "video"
        opts = vision.PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(model_path)),
            running_mode=vision.RunningMode.VIDEO if self._video else vision.RunningMode.IMAGE,
            num_poses=1,
            min_pose_detection_confidence=min_detection_confidence,
            min_pose_presence_confidence=min_presence_confidence,
            min_tracking_confidence=0.5,
        )
        self.landmarker = vision.PoseLandmarker.create_from_options(opts)
        self.margin = margin
        self.name = f"mediapipe-pose-{Path(model_path).stem}"
        self._last_ts = -1

    def estimate(self, frame: Frame, bbox: np.ndarray) -> PoseResult | None:
        import mediapipe as mp

        X1, Y1, X2, Y2 = crop_box(bbox, frame.image.shape, self.margin)
        crop = frame.image[Y1:Y2, X1:X2]
        if crop.shape[0] < 32 or crop.shape[1] < 16:
            return None
        rgb = np.ascontiguousarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        if self._video:
            ts = max(int(frame.timestamp_s * 1000), self._last_ts + 1)
            self._last_ts = ts
            res = self.landmarker.detect_for_video(image, ts)
        else:
            res = self.landmarker.detect(image)
        if not res.pose_landmarks:
            return None
        lms = res.pose_landmarks[0]
        cw, ch = X2 - X1, Y2 - Y1
        xy = np.array([[lm.x * cw + X1, lm.y * ch + Y1] for lm in lms], float)
        z = np.array([lm.z for lm in lms], float)
        vis = np.array([lm.visibility if lm.visibility is not None else 0.0 for lm in lms], float)
        pres = np.array([lm.presence if lm.presence is not None else 0.0 for lm in lms], float)
        return PoseResult(xy, z, vis, pres)

    def close(self) -> None:
        self.landmarker.close()


class TruthPoseEstimator:
    """Ground-truth landmarks of a synthetic scene plus Gaussian noise (demo/tests only)."""

    name = "synthetic-truth"

    def __init__(self, truth: dict | str | Path, noise_px: float = 1.5, seed: int = 0):
        if not isinstance(truth, dict):
            truth = json.loads(Path(truth).read_text())
        self.truth = truth
        self.times = np.array([f["t"] for f in truth["frames"]])
        self.noise = noise_px
        self.rng = np.random.default_rng(seed)

    def estimate(self, frame: Frame, bbox: np.ndarray) -> PoseResult | None:
        i = int(np.clip(np.searchsorted(self.times, frame.timestamp_s - 1e-4), 0, len(self.times) - 1))
        players = self.truth["frames"][i]["players"]
        centre = np.array([(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2]) / frame.scale
        best = min(
            players,
            key=lambda p: np.linalg.norm(
                np.array([(p["bbox"][0] + p["bbox"][2]) / 2, (p["bbox"][1] + p["bbox"][3]) / 2]) - centre
            ),
        )
        xy = np.asarray(best["landmarks"], float) * frame.scale + self.rng.normal(0, self.noise, (NUM_LANDMARKS, 2))
        vis = np.clip(0.93 + self.rng.normal(0, 0.03, NUM_LANDMARKS), 0, 1)
        return PoseResult(xy, np.zeros(NUM_LANDMARKS), vis, vis.copy())

    def close(self) -> None:
        pass
