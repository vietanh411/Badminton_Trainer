"""Person-detector interface and court-region filtering."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np

from vision.calibration.homography import project
from vision.geometry.court import COURT_LENGTH, COURT_WIDTH
from vision.video.reader import Frame


@dataclass
class Detection:
    bbox: np.ndarray  # x1, y1, x2, y2 in analysis-frame pixels
    score: float

    @property
    def foot_point(self) -> np.ndarray:
        x1, _y1, x2, y2 = self.bbox
        return np.array([(x1 + x2) / 2, y2])

    @property
    def height(self) -> float:
        return float(self.bbox[3] - self.bbox[1])


class PersonDetector(Protocol):
    name: str
    license: str

    def detect(self, frame: Frame) -> list[Detection]: ...

    def close(self) -> None: ...


def auto_device() -> str:
    """'cuda' when a CUDA GPU is usable, 'mps' on Apple silicon, else 'cpu'."""
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return "mps"
    except Exception:  # torch not installed or broken CUDA install
        pass
    return "cpu"


def filter_detections(
    dets: list[Detection],
    *,
    frame_height: int,
    H: np.ndarray | None = None,
    court_margin_m: float = 2.0,
    min_height_frac: float = 0.04,
    min_score: float = 0.0,
) -> list[Detection]:
    """Drop detections that cannot be a player on this court.

    * tiny boxes (spectators far away, false positives)
    * with a valid homography: boxes whose foot point maps outside the court plus a margin
      (umpires, coaches, people on neighbouring courts). The margin allows lunges and players
      standing behind the baseline.
    """
    out = []
    for d in dets:
        if d.score < min_score or d.height < min_height_frac * frame_height:
            continue
        if H is not None:
            x, y = project(H, d.foot_point[None])[0]
            if not np.isfinite(x) or not (
                -court_margin_m <= x <= COURT_WIDTH + court_margin_m
                and -court_margin_m - 1.0 <= y <= COURT_LENGTH + court_margin_m + 1.0
            ):
                continue
        out.append(d)
    return out
