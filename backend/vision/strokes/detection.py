"""Stroke analysis.

Layer A — **event detection** (supported): candidate hitting moments are peaks of the racket
wrist's speed, measured in torso lengths per second so the threshold does not depend on the
player's distance from the camera.

Layer B — **attributes** (experimental): two independent attributes, because "forehand/backhand"
describes *which side* the stroke is played on, while "smash/clear/drop" describe *what the shot
does*:

* ``stroke_side``: forehand | backhand | uncertain — from which side of the body the racket
  wrist is on at contact, using the shoulder line to define the player's own left/right.
* ``height_class``: overhead | side | underarm — wrist height relative to head and hips.
* ``shot_type``: drive | net_shot | lift | overhead_unspecified | unknown. Smash vs clear vs
  drop is **not** inferred: the arm action is deliberately similar (deception is the point) and
  the distinguishing evidence is shuttle speed and trajectory, which we do not track. A trained
  classifier can replace these heuristics once labelled data exists (see ``classifier.py``).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
from scipy.signal import find_peaks, savgol_filter

from vision.pose.landmarks import (
    L_ELBOW,
    L_HIP,
    L_SHOULDER,
    L_WRIST,
    NOSE,
    R_ELBOW,
    R_HIP,
    R_SHOULDER,
    R_WRIST,
    body_metrics,
)

HEURISTIC_VERSION = "heuristic-v1"


@dataclass
class StrokeConfig:
    handedness: str = "auto"  # auto | right | left
    min_peak_speed: float = 6.0  # torso lengths / s
    min_prominence: float = 3.0
    min_separation_s: float = 0.6
    window_before_s: float = 0.4
    window_after_s: float = 0.3
    min_arm_visibility: float = 0.5
    max_interp_gap_s: float = 0.15

    def to_dict(self) -> dict:
        return asdict(self)


def _interp_short_gaps(t: np.ndarray, v: np.ndarray, max_gap: float) -> np.ndarray:
    out = v.copy()
    ok = np.isfinite(v)
    if ok.sum() < 2:
        return out
    idx = np.flatnonzero(~ok)
    filled = np.interp(t[idx], t[ok], v[ok])
    prev_ok = np.searchsorted(t[ok], t[idx]) - 1
    for k, i in enumerate(idx):
        p = prev_ok[k]
        if p < 0 or p + 1 >= ok.sum():
            continue
        gap = t[ok][p + 1] - t[ok][p]
        if gap <= max_gap:
            out[i] = filled[k]
    return out


def normalised_pose(xy: np.ndarray, vis: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Landmarks relative to the hip midpoint, in torso lengths (rolling-median scale)."""
    m = body_metrics(xy, vis)
    torso = pd.Series(m["torso_length_px"]).rolling(31, center=True, min_periods=1).median().to_numpy()
    torso = pd.Series(torso).ffill().bfill().to_numpy()
    hip = (xy[:, L_HIP] + xy[:, R_HIP]) / 2
    rel = (xy - hip[:, None, :]) / torso[:, None, None]
    return rel, torso


def detect_handedness(t: np.ndarray, rel: np.ndarray, vis: np.ndarray) -> str:
    speeds = {}
    for side, wr in (("left", L_WRIST), ("right", R_WRIST)):
        p = rel[:, wr].copy()
        p[vis[:, wr] < 0.5] = np.nan
        v = np.hypot(np.gradient(p[:, 0], t), np.gradient(p[:, 1], t))
        speeds[side] = np.nanpercentile(v, 95) if np.isfinite(v).sum() > 10 else 0.0
    return "right" if speeds["right"] >= speeds["left"] else "left"


def wrist_speed(t: np.ndarray, rel: np.ndarray, vis: np.ndarray, wrist: int, cfg: StrokeConfig) -> np.ndarray:
    px = rel[:, wrist, 0].copy()
    py = rel[:, wrist, 1].copy()
    bad = vis[:, wrist] < cfg.min_arm_visibility
    px[bad] = np.nan
    py[bad] = np.nan
    px = _interp_short_gaps(t, px, cfg.max_interp_gap_s)
    py = _interp_short_gaps(t, py, cfg.max_interp_gap_s)
    if np.isfinite(px).sum() >= 7:
        ok = np.isfinite(px)
        # Light smoothing on contiguous runs to suppress landmark jitter.
        for run in np.split(np.flatnonzero(ok), np.flatnonzero(np.diff(np.flatnonzero(ok)) > 1) + 1):
            if len(run) >= 5:
                px[run] = savgol_filter(px[run], 5, 2)
                py[run] = savgol_filter(py[run], 5, 2)
    return np.hypot(np.gradient(px, t), np.gradient(py, t))


def detect_strokes(
    t: np.ndarray,
    xy: np.ndarray,
    vis: np.ndarray,
    cfg: StrokeConfig | None = None,
    court_depth_from_net: np.ndarray | None = None,
) -> tuple[list[dict], dict]:
    """Detect candidate stroke events. Returns (events, diagnostics)."""
    cfg = cfg or StrokeConfig()
    t = np.asarray(t, float)
    pose_ok = np.all(np.isfinite(xy), axis=(1, 2))
    if pose_ok.sum() < 10:
        return [], {"reason": "not enough frames with pose", "handedness": None}
    rel, _ = normalised_pose(np.where(pose_ok[:, None, None], xy, np.nan), np.where(pose_ok[:, None], vis, 0.0))
    hand = detect_handedness(t, rel, vis) if cfg.handedness == "auto" else cfg.handedness
    wr, el, sh = (R_WRIST, R_ELBOW, R_SHOULDER) if hand == "right" else (L_WRIST, L_ELBOW, L_SHOULDER)
    speed = wrist_speed(t, rel, vis, wr, cfg)
    fps = 1.0 / float(np.median(np.diff(t)))
    s = np.nan_to_num(speed, nan=0.0)
    peaks, props = find_peaks(
        s, height=cfg.min_peak_speed, prominence=cfg.min_prominence, distance=max(1, int(cfg.min_separation_s * fps))
    )
    metrics = body_metrics(np.where(pose_ok[:, None, None], xy, np.nan), vis)
    elbow = metrics[f"{hand}_elbow_angle"]
    events = []
    for k, p in enumerate(peaks):
        lo = np.searchsorted(t, t[p] - cfg.window_before_s)
        hi = np.searchsorted(t, t[p] + cfg.window_after_s, side="right")
        # Contact ≈ highest-speed sample; the wrist is usually fastest just before/at contact.
        arm_vis = float(np.nanmean(vis[lo:hi, [sh, el, wr]]))
        prom = float(props["prominences"][k])
        conf = float(np.clip((1 - np.exp(-prom / 6.0)) * np.clip(arm_vis, 0, 1) * min(1.0, fps / 25.0) ** 0.5, 0, 1))
        attrs = heuristic_attributes(
            rel, vis, metrics, p, lo, hi, hand, None if court_depth_from_net is None else court_depth_from_net[p]
        )
        el_win = elbow[lo:hi]
        events.append(
            {
                "t": round(float(t[p]), 3),
                "t_start": round(float(t[lo]), 3),
                "t_end": round(float(t[min(hi, len(t)) - 1]), 3),
                "sample_index": int(p),
                "confidence": round(conf, 3),
                "signals": {
                    "peak_wrist_speed_torso_per_s": round(float(s[p]), 2),
                    "prominence": round(prom, 2),
                    "wrist_height_above_shoulder_torso": round(float(-(rel[p, wr, 1] - rel[p, sh, 1])), 3)
                    if np.isfinite(rel[p, wr, 1])
                    else None,
                    "elbow_angle_at_peak_deg": None if not np.isfinite(elbow[p]) else round(float(elbow[p]), 1),
                    "elbow_angle_range_deg": None
                    if not np.isfinite(el_win).any()
                    else round(float(np.nanmax(el_win) - np.nanmin(el_win)), 1),
                    "arm_visibility": round(arm_vis, 3),
                },
                **attrs,
            }
        )
    diag = {
        "handedness": hand,
        "sampling_fps": round(fps, 2),
        "candidates": len(events),
        "wrist_speed_p95": float(np.nanpercentile(speed, 95)) if np.isfinite(speed).any() else None,
        "warning": "sampling below 25 fps under-resolves fast swings; enable a higher analysis FPS"
        if fps < 25
        else None,
    }
    return events, diag


def heuristic_attributes(rel, vis, metrics, p, lo, hi, hand, depth_from_net) -> dict:
    wr = R_WRIST if hand == "right" else L_WRIST
    # Height class from the highest wrist position just before/at contact (image y is down; rel
    # is hip-centred). The follow-through is excluded: a lift finishes high but is struck low.
    win = slice(max(lo, p - 2), p + 1)
    wrist_y = np.nanmin(rel[win, wr, 1]) if np.isfinite(rel[win, wr, 1]).any() else np.nan
    nose_y = np.nanmedian(rel[win, NOSE, 1])
    hip_y = 0.0
    if not np.isfinite(wrist_y):
        height = "unknown"
    elif np.isfinite(nose_y) and wrist_y < nose_y:
        height = "overhead"
    elif rel[p, wr, 1] > hip_y - 0.1:
        height = "underarm"
    else:
        height = "side"

    # Side: project wrist offset onto the player's own left→right shoulder axis.
    sh_vec = rel[p, R_SHOULDER] - rel[p, L_SHOULDER]
    sh_mid = (rel[p, R_SHOULDER] + rel[p, L_SHOULDER]) / 2
    side_on = metrics["torso_orientation"][p]
    stroke_side, side_conf = "uncertain", 0.0
    if np.all(np.isfinite(sh_vec)) and np.linalg.norm(sh_vec) > 0.2 and np.isfinite(rel[p, wr]).all():
        lateral = float(np.dot(rel[p, wr] - sh_mid, sh_vec / np.linalg.norm(sh_vec)))
        toward_racket_side = lateral if hand == "right" else -lateral
        margin = abs(toward_racket_side)
        if margin > 0.25 and (not np.isfinite(side_on) or side_on > 0.35):
            stroke_side = "forehand" if toward_racket_side > 0 else "backhand"
            side_conf = float(np.clip(margin / 1.0, 0.2, 0.8))

    if height == "overhead":
        shot, shot_conf = "overhead_unspecified", 0.5
    elif height == "side":
        shot, shot_conf = "drive", 0.35
    elif height == "underarm":
        if depth_from_net is not None and np.isfinite(depth_from_net):
            shot, shot_conf = ("net_shot", 0.35) if depth_from_net < 2.4 else ("lift", 0.3)
        else:
            shot, shot_conf = "underarm_unspecified", 0.3
    else:
        shot, shot_conf = "unknown", 0.0
    return {
        "height_class": height,
        "stroke_side": stroke_side,
        "stroke_side_confidence": round(side_conf, 3),
        "shot_type": shot,
        "shot_type_confidence": round(shot_conf, 3),
        "classifier": HEURISTIC_VERSION,
        "experimental": True,
    }
