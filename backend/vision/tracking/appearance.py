"""Cheap appearance descriptor for re-linking a player's track after occlusions.

A hue/saturation histogram of the upper body (shirt) is robust to pose changes and costs
microseconds. It is not a learned re-ID embedding; it only needs to tell two players apart on
one court, which usually wear different colours. When they do not, linking falls back to motion.
"""

from __future__ import annotations

import cv2
import numpy as np

HIST_BINS = (8, 4)
HIST_DIM = HIST_BINS[0] * HIST_BINS[1]


def appearance_descriptor(image: np.ndarray, bbox: np.ndarray) -> np.ndarray:
    h_img, w_img = image.shape[:2]
    x1, y1, x2, y2 = bbox
    bw, bh = x2 - x1, y2 - y1
    rx1 = int(np.clip(x1 + 0.2 * bw, 0, w_img - 1))
    rx2 = int(np.clip(x2 - 0.2 * bw, rx1 + 1, w_img))
    ry1 = int(np.clip(y1 + 0.15 * bh, 0, h_img - 1))
    ry2 = int(np.clip(y1 + 0.55 * bh, ry1 + 1, h_img))
    crop = image[ry1:ry2, rx1:rx2]
    if crop.size == 0:
        return np.full(HIST_DIM, 1.0 / HIST_DIM, np.float32)
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, list(HIST_BINS), [0, 180, 0, 256]).ravel().astype(np.float32)
    s = hist.sum()
    return hist / s if s > 0 else np.full(HIST_DIM, 1.0 / HIST_DIM, np.float32)


def appearance_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Bhattacharyya distance in [0, 1] between two normalised histograms."""
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    if not (np.isfinite(a).all() and np.isfinite(b).all()) or a.sum() <= 0 or b.sum() <= 0:
        return 1.0
    bc = np.sum(np.sqrt(a / a.sum() * b / b.sum()))
    return float(np.sqrt(max(0.0, 1.0 - bc)))
