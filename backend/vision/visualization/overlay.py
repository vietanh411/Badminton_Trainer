"""Burn analysis overlays into video frames (annotated clip export)."""

from __future__ import annotations

import threading
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from vision.calibration.homography import project
from vision.geometry.court import court_lines
from vision.pose.landmarks import SKELETON
from vision.video.reader import iter_frames
from vision.video.writer import open_writer

CYAN = (238, 211, 34)  # BGR of #22d3ee
LIME = (53, 230, 163)
GREY = (200, 200, 200)
ORANGE = (22, 115, 249)


def court_polylines(H_inv: np.ndarray, n: int = 40) -> list[np.ndarray]:
    out = []
    for ln in court_lines():
        t = np.linspace(0, 1, n)[:, None]
        seg = np.asarray(ln.start, float) + t * (np.asarray(ln.end, float) - np.asarray(ln.start, float))
        uv = project(H_inv, seg)
        ok = np.isfinite(uv).all(axis=1)
        if ok.sum() >= 2:
            out.append(uv[ok].astype(np.int32))
    return out


def render_annotated_clip(
    video: Path,
    stem: Path,
    positions: pd.DataFrame,
    tracks: pd.DataFrame,
    pose_xy: np.ndarray,
    pose_vis: np.ndarray,
    analysis_fps: float,
    max_width: int,
    H_analysis: np.ndarray | None,
    start: float,
    end: float,
    synthetic: bool,
    cancel: threading.Event | None = None,
) -> Path:
    """Render [start, end] at the analysis frame rate with court lines, boxes, skeleton and trail."""
    pos = positions.reset_index(drop=True)
    by_sample = {int(si): i for i, si in enumerate(pos["sample_index"])}
    tracks_by = {int(k): g for k, g in tracks.groupby("sample_index")}
    lines = court_polylines(np.linalg.inv(H_analysis)) if H_analysis is not None else []
    writer = path = None
    trail: list[tuple[int, int]] = []
    try:
        for frame in iter_frames(video, target_fps=analysis_fps, max_width=max_width, cancel=cancel):
            if frame.timestamp_s < start - 1e-6:
                continue
            if frame.timestamp_s > end:
                break
            img = frame.image.copy()
            if writer is None:
                h, w = img.shape[:2]
                writer, path, _ = open_writer(stem, analysis_fps, (w, h))
            for pl in lines:
                cv2.polylines(img, [pl], False, GREY, 1, cv2.LINE_AA)
            i = by_sample.get(frame.sample_index)
            target = None if i is None or pd.isna(pos.at[i, "track_id"]) else int(pos.at[i, "track_id"])
            for r in tracks_by.get(frame.sample_index, pd.DataFrame()).itertuples():
                is_t = target is not None and int(r.track_id) == target
                color = CYAN if is_t else GREY
                cv2.rectangle(img, (int(r.x1), int(r.y1)), (int(r.x2), int(r.y2)), color, 2 if is_t else 1, cv2.LINE_AA)
                cv2.putText(
                    img,
                    f"#{int(r.track_id)}",
                    (int(r.x1), int(r.y1) - 4),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    color,
                    1,
                    cv2.LINE_AA,
                )
            if i is not None:
                xy = pose_xy[i]
                vis = pose_vis[i]
                if np.all(np.isfinite(xy)):
                    for a, b in SKELETON:
                        if vis[a] > 0.5 and vis[b] > 0.5:
                            cv2.line(
                                img,
                                tuple(np.round(xy[a]).astype(int)),
                                tuple(np.round(xy[b]).astype(int)),
                                LIME,
                                2,
                                cv2.LINE_AA,
                            )
                if np.isfinite(pos.at[i, "img_x"]):
                    p = (int(pos.at[i, "img_x"]), int(pos.at[i, "img_y"]))
                    trail.append(p)
                    trail = trail[-45:]
                    cv2.circle(img, p, 5, ORANGE, -1, cv2.LINE_AA)
                if "speed" in pos and np.isfinite(pos.at[i, "speed"]):
                    cv2.putText(
                        img,
                        f"{pos.at[i, 'speed']:.1f} m/s",
                        (12, 28),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        CYAN,
                        2,
                        cv2.LINE_AA,
                    )
            if len(trail) >= 2:
                cv2.polylines(img, [np.array(trail, np.int32)], False, ORANGE, 2, cv2.LINE_AA)
            cv2.putText(
                img,
                f"t={frame.timestamp_s:6.2f}s",
                (12, img.shape[0] - 12),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                GREY,
                1,
                cv2.LINE_AA,
            )
            if synthetic:
                cv2.putText(
                    img,
                    "SYNTHETIC DEMO DATA",
                    (img.shape[1] - 230, 28),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 140, 255),
                    2,
                    cv2.LINE_AA,
                )
            writer.write(img)
    finally:
        if writer is not None:
            writer.release()
    if path is None:
        raise ValueError("no frames in the requested interval")
    return path
