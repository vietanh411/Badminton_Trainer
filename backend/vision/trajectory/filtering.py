"""Turn raw per-frame court positions into a clean, honest trajectory.

Raw and filtered coordinates are kept side by side. Every sample carries flags explaining what
happened to it, so downstream metrics (and users) can tell measured, interpolated, rejected and
missing data apart.

Pipeline: bounds check → robust outlier rejection (rolling median residual) → physical speed
gate → segmenting at long gaps → linear interpolation of short gaps only → Savitzky–Golay
smoothing per segment → velocity / acceleration by finite differences on real timestamps.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
from scipy.ndimage import median_filter
from scipy.signal import savgol_filter

from vision.geometry.court import COURT_LENGTH, COURT_WIDTH


@dataclass
class TrajectoryConfig:
    court_margin_m: float = 1.5  # positions further outside the doubles court are impossible
    min_confidence: float = 0.25
    outlier_window_s: float = 0.6
    outlier_min_residual_m: float = 0.6
    max_speed_mps: float = 8.0  # elite badminton movement peaks around 5–6 m/s
    speed_gate_slack_m: float = 0.25
    max_interp_gap_s: float = 0.4
    # "kalman": constant-velocity Kalman filter + Rauch–Tung–Striebel smoother (handles uneven time
    # steps, weights samples by confidence, estimates velocity directly). "savgol": Savitzky–Golay.
    smoother: str = "kalman"
    kalman_measurement_sigma_m: float = 0.12  # foot-position noise at confidence 1
    kalman_accel_density: float = 30.0  # white-noise acceleration spectral density q (m²/s³)
    smoothing_window_s: float = 0.5  # Savitzky–Golay window; 0 disables smoothing
    smoothing_polyorder: int = 2

    def to_dict(self) -> dict:
        return asdict(self)


def _odd(n: int) -> int:
    return n if n % 2 == 1 else n + 1


def _segments(valid: np.ndarray, t: np.ndarray, max_gap: float, breaks: np.ndarray | None = None) -> np.ndarray:
    """Segment id per sample: consecutive valid samples whose gaps are ≤ max_gap share an id.

    ``breaks[i]`` forces a new segment to start at sample i (an accepted position jump).
    """
    seg = np.full(len(valid), -1)
    sid = -1
    last_t = None
    for i in np.flatnonzero(valid):
        if last_t is None or t[i] - last_t > max_gap + 1e-9 or (breaks is not None and breaks[i]):
            sid += 1
        seg[i] = sid
        last_t = t[i]
    return seg


def process_trajectory(
    t: np.ndarray,
    x_raw: np.ndarray,
    y_raw: np.ndarray,
    conf: np.ndarray,
    cfg: TrajectoryConfig | None = None,
    breaks: np.ndarray | None = None,
) -> pd.DataFrame:
    """Filter a trajectory. ``breaks`` (e.g. a shot id per sample) marks hard discontinuities such as
    video cuts: each block is processed on its own, so no filter, interpolation, segment or derivative
    ever spans a cut."""
    if breaks is not None:
        breaks = np.asarray(breaks)
        starts = np.flatnonzero(np.r_[True, breaks[1:] != breaks[:-1]])
        ends = np.r_[starts[1:], len(breaks)]
        if len(starts) > 1:
            parts = []
            offset = 0
            for a, b in zip(starts, ends, strict=True):
                part = process_trajectory(t[a:b], x_raw[a:b], y_raw[a:b], conf[a:b], cfg)
                seg = part["segment"].to_numpy()
                part["segment"] = np.where(seg >= 0, seg + offset, -1)
                offset = max(offset, int(part["segment"].max()) + 1)
                parts.append(part)
            return pd.concat(parts, ignore_index=True)
    cfg = cfg or TrajectoryConfig()
    t = np.asarray(t, float)
    x_raw = np.asarray(x_raw, float)
    y_raw = np.asarray(y_raw, float)
    conf = np.asarray(conf, float)
    n = len(t)
    reason = np.array([""] * n, dtype=object)

    finite = np.isfinite(x_raw) & np.isfinite(y_raw)
    reason[~finite] = "missing"
    low_conf = finite & (conf < cfg.min_confidence)
    reason[low_conf] = "low_confidence"
    m = cfg.court_margin_m
    oob = finite & ~low_conf & ((x_raw < -m) | (x_raw > COURT_WIDTH + m) | (y_raw < -m) | (y_raw > COURT_LENGTH + m))
    reason[oob] = "out_of_bounds"
    valid = finite & ~low_conf & ~oob

    dt_med = float(np.median(np.diff(t))) if n > 1 else 1 / 30
    fps = 1.0 / max(dt_med, 1e-3)

    # Robust outlier rejection: residual from a rolling median of the valid points.
    idx = np.flatnonzero(valid)
    if len(idx) >= 5:
        w = _odd(max(3, int(round(cfg.outlier_window_s * fps))))
        mx = median_filter(x_raw[idx], size=w, mode="nearest")
        my = median_filter(y_raw[idx], size=w, mode="nearest")
        res = np.hypot(x_raw[idx] - mx, y_raw[idx] - my)
        mad = np.median(np.abs(res - np.median(res))) * 1.4826
        thr = max(cfg.outlier_min_residual_m, 5.0 * mad)
        bad = idx[res > thr]
        valid[bad] = False
        reason[bad] = "outlier"

    # Physical speed gate relative to the last accepted point. If the new location persists (more than
    # 3 consecutive violations) it is accepted, but as the start of a new segment: whatever caused the
    # jump (a missed identity swap, a pose glitch), the path between the two places was not observed.
    last = None
    rejected_run = 0
    jump_break = np.zeros(n, bool)
    for i in np.flatnonzero(valid):
        if last is None:
            last = i
            continue
        dt = t[i] - t[last]
        d = np.hypot(x_raw[i] - x_raw[last], y_raw[i] - y_raw[last])
        if dt > cfg.max_interp_gap_s * 2:  # after a long gap anything goes; start fresh
            last = i
            rejected_run = 0
            continue
        too_fast = d > cfg.max_speed_mps * dt + cfg.speed_gate_slack_m
        if too_fast and rejected_run < 3:
            valid[i] = False
            reason[i] = "speed_gate"
            rejected_run += 1
            continue
        if too_fast:
            jump_break[i] = True
        last = i
        rejected_run = 0

    # Segments, interpolation of short gaps.
    seg_valid = _segments(valid, t, cfg.max_interp_gap_s, jump_break)
    x = np.full(n, np.nan)
    y = np.full(n, np.nan)
    interpolated = np.zeros(n, bool)
    segment = np.full(n, -1)
    kalman_v = np.full((n, 2), np.nan)
    for sid in np.unique(seg_valid[seg_valid >= 0]):
        members = np.flatnonzero(seg_valid == sid)
        a, b = members[0], members[-1]
        span = np.arange(a, b + 1)
        x[span] = np.interp(t[span], t[members], x_raw[members])
        y[span] = np.interp(t[span], t[members], y_raw[members])
        interpolated[span] = ~np.isin(span, members)
        segment[span] = sid
        # Smoothing.
        if cfg.smoother == "kalman" and len(members) >= 2:
            zx = np.full(len(span), np.nan)
            zy = np.full(len(span), np.nan)
            rel = members - a
            zx[rel] = x_raw[members]
            zy[rel] = y_raw[members]
            sig = cfg.kalman_measurement_sigma_m / np.clip(conf[span], 0.2, 1.0)
            x[span], kvx = rts_smooth(t[span], zx, sig, cfg.kalman_accel_density)
            y[span], kvy = rts_smooth(t[span], zy, sig, cfg.kalman_accel_density)
            kalman_v[span, 0] = kvx
            kalman_v[span, 1] = kvy
        elif cfg.smoother != "kalman" and cfg.smoothing_window_s > 0 and len(span) >= 5:
            w = _odd(max(5, int(round(cfg.smoothing_window_s * fps))))
            w = min(w, len(span) if len(span) % 2 == 1 else len(span) - 1)
            if w > cfg.smoothing_polyorder + 1:
                x[span] = savgol_filter(x[span], w, cfg.smoothing_polyorder, mode="interp")
                y[span] = savgol_filter(y[span], w, cfg.smoothing_polyorder, mode="interp")

    vx = np.full(n, np.nan)
    vy = np.full(n, np.nan)
    ax = np.full(n, np.nan)
    ay = np.full(n, np.nan)
    for sid in np.unique(segment[segment >= 0]):
        span = np.flatnonzero(segment == sid)
        if len(span) < 3:
            continue
        ts = t[span]
        if cfg.smoother == "kalman" and np.isfinite(kalman_v[span]).all():
            vx[span] = kalman_v[span, 0]
            vy[span] = kalman_v[span, 1]
        else:
            vx[span] = np.gradient(x[span], ts)
            vy[span] = np.gradient(y[span], ts)
        ax[span] = np.gradient(vx[span], ts)
        ay[span] = np.gradient(vy[span], ts)
    speed = np.hypot(vx, vy)
    # Tangential acceleration (+ speeding up, − slowing down).
    with np.errstate(invalid="ignore", divide="ignore"):
        tang = (vx * ax + vy * ay) / np.where(speed > 1e-6, speed, np.nan)

    return pd.DataFrame(
        {
            "t": t,
            "court_x_raw": x_raw,
            "court_y_raw": y_raw,
            "position_confidence": conf,
            "raw_valid": valid,
            "reject_reason": reason,
            "court_x": x,
            "court_y": y,
            "interpolated": interpolated,
            "segment": segment,
            "vx": vx,
            "vy": vy,
            "speed": speed,
            "accel": np.hypot(ax, ay),
            "tangential_accel": tang,
        }
    )


def sample_weights(t: np.ndarray, valid: np.ndarray, segment: np.ndarray) -> np.ndarray:
    """Seconds of observed time represented by each sample.

    Each valid sample stands for the interval until the next sample *in the same segment*
    (capped at 1.5× the median frame interval), so dropped frames, irregular timestamps and
    gaps never inflate occupancy. Samples outside any segment weigh 0.
    """
    t = np.asarray(t, float)
    n = len(t)
    w = np.zeros(n)
    if n < 2:
        return w
    dt_med = float(np.median(np.diff(t)))
    cap = 1.5 * dt_med
    for sid in np.unique(segment[segment >= 0]):
        span = np.flatnonzero((segment == sid) & valid)
        if len(span) == 0:
            continue
        d = np.diff(t[span], append=t[span[-1]] + dt_med)
        w[span] = np.minimum(d, cap)
    return w


def rts_smooth(t: np.ndarray, z: np.ndarray, sigma: np.ndarray, q: float) -> tuple[np.ndarray, np.ndarray]:
    """Constant-velocity Kalman filter + Rauch–Tung–Striebel smoother for one coordinate.

    ``z`` may contain NaN (no measurement: the smoother bridges it); ``sigma`` is the per-sample
    measurement standard deviation; ``q`` the white-noise acceleration spectral density. Time steps
    may be uneven. Returns (smoothed position, smoothed velocity).
    """
    n = len(t)
    xs = np.zeros((n, 2))
    Ps = np.zeros((n, 2, 2))
    xp = np.zeros((n, 2))
    Pp = np.zeros((n, 2, 2))
    first = int(np.flatnonzero(np.isfinite(z))[0])
    x = np.array([z[first], 0.0])
    P = np.diag([sigma[first] ** 2, 9.0])
    Hm = np.array([[1.0, 0.0]])
    for k in range(n):
        if k > 0:
            dt = max(t[k] - t[k - 1], 1e-4)
            F = np.array([[1.0, dt], [0.0, 1.0]])
            Q = q * np.array([[dt**3 / 3, dt**2 / 2], [dt**2 / 2, dt]])
            x = F @ x
            P = F @ P @ F.T + Q
        xp[k], Pp[k] = x, P
        if np.isfinite(z[k]):
            S = P[0, 0] + sigma[k] ** 2
            K = P[:, 0] / S
            x = x + K * (z[k] - x[0])
            P = P - np.outer(K, Hm @ P)
        xs[k], Ps[k] = x, P
    for k in range(n - 2, -1, -1):
        dt = max(t[k + 1] - t[k], 1e-4)
        F = np.array([[1.0, dt], [0.0, 1.0]])
        C = Ps[k] @ F.T @ np.linalg.inv(Pp[k + 1])
        xs[k] = xs[k] + C @ (xs[k + 1] - xp[k + 1])
        Ps[k] = Ps[k] + C @ (Ps[k + 1] - Pp[k + 1]) @ C.T
    return xs[:, 0], xs[:, 1]
