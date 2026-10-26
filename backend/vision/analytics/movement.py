"""Movement metrics computed from the filtered court trajectory.

Every metric has a written definition in ``docs/methodology.md``. Conventions:

* time-based quantities use per-sample weights from :func:`sample_weights` (real seconds of
  observation), so results are comparable across frame rates and robust to dropped frames;
* nothing is computed across a tracking gap: distance, speed and direction changes only use
  consecutive samples inside one trajectory segment;
* recovery and footwork values are **movement proxies**. Without shuttle tracking we do not
  know when shots happen, so "recovery" means "returning towards the base position".
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np
import pandas as pd

from vision.geometry.court import NET_Y, ZoneConfig, assign_zones, from_half_coords, to_half_coords
from vision.trajectory.filtering import sample_weights


@dataclass
class MovementConfig:
    moving_on_mps: float = 0.6  # hysteresis: start "moving" above this speed
    moving_off_mps: float = 0.3  # ... and stop below this one
    direction_min_speed_mps: float = 0.8
    direction_min_angle_deg: float = 60.0
    direction_min_displacement_m: float = 0.4
    direction_persist_s: float = 0.12
    base_lateral_m: float = 2.59  # player-relative base position (half coordinates)
    base_depth_m: float = 3.3  # distance from the net
    far_from_base_m: float = 1.5
    returned_to_base_m: float = 0.8
    zone_visit_min_s: float = 0.2
    zone_frequent_share: float = 0.05
    speed_hist_bin_mps: float = 0.5
    speed_hist_max_mps: float = 8.0
    composite_enabled: bool = True
    composite_weights: dict[str, float] = field(
        default_factory=lambda: {"directness": 0.3, "recovery": 0.3, "smoothness": 0.2, "consistency": 0.2}
    )

    def to_dict(self) -> dict:
        return asdict(self)


def _hysteresis(speed: np.ndarray, on: float, off: float) -> np.ndarray:
    moving = np.zeros(len(speed), bool)
    state = False
    for i, s in enumerate(speed):
        if not np.isfinite(s):
            state = False
            continue
        if state and s < off:
            state = False
        elif not state and s > on:
            state = True
        moving[i] = state
    return moving


def player_side(y: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Side (0 near, 1 far) per sample, smoothed so a lunge to the net does not flip sides."""
    side = (np.asarray(y) >= NET_Y).astype(float)
    side[~np.isfinite(y)] = np.nan
    s = pd.Series(side).rolling(61, center=True, min_periods=1).median().to_numpy()
    s = pd.Series(s).ffill().bfill().to_numpy()
    if np.all(np.isnan(s)):
        return np.zeros(len(y), int)
    return np.round(s).astype(int)


def direction_changes(t, x, y, vx, vy, segment, cfg: MovementConfig) -> list[dict]:
    """Meaningful changes of movement heading.

    A heading is "established" once the player moves faster than ``direction_min_speed_mps``
    in a consistent direction for ``direction_persist_s``. A change is counted when a new
    established heading differs from the previous one by ≥ ``direction_min_angle_deg`` *and*
    the player travelled ≥ ``direction_min_displacement_m`` since the previous change.
    Noise-level jitter never reaches the speed threshold, and brief wobbles fail the
    persistence requirement (hysteresis).
    """
    events = []
    speed = np.hypot(vx, vy)
    heading = np.degrees(np.arctan2(vy, vx))
    prev_heading = None
    prev_pos = None
    cand = None  # (start index, heading)
    last_seg = None
    for i in range(len(t)):
        if segment[i] < 0 or not np.isfinite(speed[i]):
            cand = None
            continue
        if segment[i] != last_seg:
            prev_heading, prev_pos, cand = None, None, None
            last_seg = segment[i]
        if speed[i] < cfg.direction_min_speed_mps:
            cand = None
            continue
        h = heading[i]
        if cand is None or abs((h - cand[1] + 180) % 360 - 180) > 30:
            cand = (i, h)
            continue
        if t[i] - t[cand[0]] < cfg.direction_persist_s:
            continue
        established = cand[1]
        if prev_heading is None:
            prev_heading, prev_pos = established, np.array([x[cand[0]], y[cand[0]]])
            continue
        diff = abs((established - prev_heading + 180) % 360 - 180)
        if diff >= cfg.direction_min_angle_deg:
            pos = np.array([x[cand[0]], y[cand[0]]])
            if np.linalg.norm(pos - prev_pos) >= cfg.direction_min_displacement_m:
                events.append(
                    {
                        "type": "direction_change",
                        "t": round(float(t[cand[0]]), 3),
                        "angle_deg": round(float(diff), 1),
                        "court_x": round(float(pos[0]), 3),
                        "court_y": round(float(pos[1]), 3),
                    }
                )
                prev_pos = pos
            prev_heading = established
        else:
            prev_heading = 0.8 * prev_heading + 0.2 * established if diff < 20 else prev_heading
    return events


def _ldlj(speed: np.ndarray, t: np.ndarray) -> float:
    """Log dimensionless jerk of a speed profile (Balasubramanian et al., 2015).

    Closer to zero = smoother. Sensitive to noise and sampling rate, hence only a proxy.
    """
    if len(speed) < 7:
        return float("nan")
    T = t[-1] - t[0]
    vpk = np.nanmax(speed)
    if T <= 0 or vpk <= 0:
        return float("nan")
    jerk = np.gradient(np.gradient(speed, t), t)
    integral = np.trapezoid(jerk**2, t)
    if integral <= 0:
        return float("nan")
    return float(-np.log(T**3 / vpk**2 * integral))


def compute_movement_metrics(
    traj: pd.DataFrame, cfg: MovementConfig | None = None, zone_cfg: ZoneConfig | None = None
) -> tuple[dict, list[dict], pd.DataFrame]:
    """Returns (metrics dict, movement events, per-sample derived columns)."""
    cfg = cfg or MovementConfig()
    zone_cfg = zone_cfg or ZoneConfig()
    t = traj["t"].to_numpy(float)
    x = traj["court_x"].to_numpy(float)
    y = traj["court_y"].to_numpy(float)
    seg = traj["segment"].to_numpy(int)
    speed = traj["speed"].to_numpy(float)
    vx = traj["vx"].to_numpy(float)
    vy = traj["vy"].to_numpy(float)
    tang = traj["tangential_accel"].to_numpy(float)
    has_pos = np.isfinite(x) & np.isfinite(y) & (seg >= 0)
    w = sample_weights(t, has_pos, seg)
    observed_s = float(w.sum())
    # Duration covered by the samples: last sample also represents one frame interval.
    duration_s = float(t[-1] - t[0] + np.median(np.diff(t))) if len(t) > 1 else 0.0

    # --- distance (consecutive samples within one segment only) -------------------------
    same = (seg[1:] == seg[:-1]) & (seg[1:] >= 0) & has_pos[1:] & has_pos[:-1]
    steps = np.hypot(np.diff(x), np.diff(y))
    steps = np.where(same, steps, 0.0)
    interp = traj["interpolated"].to_numpy(bool)
    total_distance = float(np.nansum(steps))
    interp_distance = float(np.nansum(np.where(interp[1:] | interp[:-1], steps, 0.0)))

    # --- speed --------------------------------------------------------------------------
    ok_speed = has_pos & np.isfinite(speed)
    moving = _hysteresis(np.where(ok_speed, speed, np.nan), cfg.moving_on_mps, cfg.moving_off_mps)
    w_speed = np.where(ok_speed, w, 0.0)
    mean_speed = (
        float(np.sum(speed[ok_speed] * w_speed[ok_speed]) / max(w_speed.sum(), 1e-9)) if ok_speed.any() else None
    )
    mw = w_speed * moving
    mean_moving_speed = float(np.sum(np.nan_to_num(speed) * mw) / mw.sum()) if mw.sum() > 0 else None
    bins = np.arange(0, cfg.speed_hist_max_mps + cfg.speed_hist_bin_mps, cfg.speed_hist_bin_mps)
    hist, _ = np.histogram(
        np.clip(speed[ok_speed], 0, cfg.speed_hist_max_mps - 1e-9), bins=bins, weights=w_speed[ok_speed]
    )
    time_moving = float(mw.sum())
    p95 = float(np.percentile(speed[ok_speed], 95)) if ok_speed.sum() > 5 else None

    # --- zones (player-relative half coordinates, side resolved per sample) -------------
    side = player_side(np.where(has_pos, y, np.nan), w)
    lat = np.full(len(t), np.nan)
    dep = np.full(len(t), np.nan)
    for s in (0, 1):
        m = has_pos & (side == s)
        lat[m], dep[m] = to_half_coords(x[m], y[m], s)
    zones = assign_zones(lat, dep, zone_cfg)
    zone_time = {z: 0.0 for z in zone_cfg.zone_ids}
    zone_entries = {z: 0 for z in zone_cfg.zone_ids}
    prev = None
    for i in np.flatnonzero(has_pos):
        z = zones[i]
        if z in zone_time:
            zone_time[z] += w[i]
            if z != prev:
                zone_entries[z] += 1
        prev = z
    out_time = float(sum(w[i] for i in np.flatnonzero(has_pos) if zones[i] == "out"))
    in_time = sum(zone_time.values())
    zone_share = {z: (v / in_time if in_time > 0 else 0.0) for z, v in zone_time.items()}
    visited = [z for z, v in zone_time.items() if v >= cfg.zone_visit_min_s]
    frequented = [z for z, s in zone_share.items() if s >= cfg.zone_frequent_share]
    depth_share = {d: sum(zone_share[f"{d}_{lt}"] for lt in zone_cfg.lateral_names) for d in zone_cfg.depth_names}
    lateral_share = {lt: sum(zone_share[f"{d}_{lt}"] for d in zone_cfg.depth_names) for lt in zone_cfg.lateral_names}

    # Convex-hull area of positions (simple coverage proxy).
    hull_area = None
    if has_pos.sum() >= 3:
        from shapely.geometry import MultiPoint

        hull_area = float(MultiPoint(np.column_stack([x[has_pos], y[has_pos]])).convex_hull.area)

    # --- recovery proxies ----------------------------------------------------------------
    base_xy = np.full((len(t), 2), np.nan)
    for s in (0, 1):
        bx, by = from_half_coords(np.array([cfg.base_lateral_m]), np.array([cfg.base_depth_m]), s)
        base_xy[side == s] = [bx[0], by[0]]
    d_base = np.hypot(x - base_xy[:, 0], y - base_xy[:, 1])
    d_base[~has_pos] = np.nan
    excursions = _excursions(t, x, y, d_base, speed, tang, seg, cfg)
    rec_times = [e["recovery_time_s"] for e in excursions if e["recovery_time_s"] is not None]
    directness = [e["directness"] for e in excursions if e["directness"] is not None]
    smooth = [e["smoothness_ldlj"] for e in excursions if e["smoothness_ldlj"] is not None]
    far_time = float(np.sum(w[np.nan_to_num(d_base) > cfg.far_from_base_m]))

    # Positioning consistency: spread of "ready" positions (stationary near base).
    ready = has_pos & ~moving & (np.nan_to_num(d_base, nan=99) < cfg.far_from_base_m)
    consistency = None
    if ready.sum() >= 5:
        pts = np.column_stack([lat[ready], dep[ready]])
        c = np.average(pts, axis=0, weights=np.maximum(w[ready], 1e-9))
        consistency = float(np.average(np.linalg.norm(pts - c, axis=1), weights=np.maximum(w[ready], 1e-9)))

    events = direction_changes(t, x, y, vx, vy, seg, cfg)
    for i0, i1 in _runs(speed > 4.0):
        events.append(
            {
                "type": "high_speed",
                "t": round(float(t[i0]), 3),
                "t_end": round(float(t[i1]), 3),
                "peak_speed_mps": round(float(np.nanmax(speed[i0 : i1 + 1])), 2),
            }
        )

    accel_ok = has_pos & np.isfinite(tang)
    footwork = {
        "path_directness_mean": _mean(directness),
        "extra_distance_pct_mean": _mean(
            [e["extra_distance_pct"] for e in excursions if e["extra_distance_pct"] is not None]
        ),
        "recovery_time_median_s": _median(rec_times),
        "peak_acceleration_p95_mps2": float(np.percentile(tang[accel_ok], 95)) if accel_ok.sum() > 5 else None,
        "peak_deceleration_p95_mps2": float(-np.percentile(tang[accel_ok], 5)) if accel_ok.sum() > 5 else None,
        "smoothness_ldlj_median": _median(smooth),
        "positioning_consistency_m": consistency,
    }
    composite = _composite(footwork, cfg) if cfg.composite_enabled else None

    valid_raw = traj["raw_valid"].to_numpy(bool)
    metrics = {
        "duration_s": duration_s,
        "observed_s": observed_s,
        "coverage_pct": 100.0 * observed_s / duration_s if duration_s > 0 else 0.0,
        "distance": {
            "total_m": total_distance,
            "interpolated_m": interp_distance,
            "per_minute_m": total_distance / (observed_s / 60) if observed_s > 0 else None,
        },
        "speed": {
            "mean_mps": mean_speed,
            "mean_while_moving_mps": mean_moving_speed,
            "p95_mps": p95,
            "max_filtered_mps": float(np.nanmax(speed[ok_speed])) if ok_speed.any() else None,
            "time_moving_s": time_moving,
            "time_stationary_s": float(w_speed.sum() - time_moving),
            "moving_share": time_moving / w_speed.sum() if w_speed.sum() > 0 else None,
            "histogram": {"bin_edges_mps": bins.tolist(), "seconds": hist.tolist()},
        },
        "zones": {
            "time_s": zone_time,
            "share": zone_share,
            "entries": zone_entries,
            "outside_singles_s": out_time,
            "visited": visited,
            "visited_pct": 100.0 * len(visited) / len(zone_cfg.zone_ids),
            "frequented": frequented,
            "depth_share": depth_share,
            "lateral_share": lateral_share,
            "convex_hull_area_m2": hull_area,
        },
        "recovery": {
            "base_position_half": {"lateral_m": cfg.base_lateral_m, "depth_from_net_m": cfg.base_depth_m},
            "mean_distance_from_base_m": float(np.nanmean(d_base)) if has_pos.any() else None,
            "time_far_from_base_s": far_time,
            "far_share": far_time / observed_s if observed_s > 0 else None,
            "excursions": len(excursions),
            "recovery_time_median_s": _median(rec_times),
            "label": "movement proxy (no shot timing available)",
        },
        "footwork": footwork,
        "composite": composite,
        "direction_changes": sum(1 for e in events if e["type"] == "direction_change"),
        "direction_changes_per_minute": (sum(1 for e in events if e["type"] == "direction_change") / (observed_s / 60))
        if observed_s > 0
        else None,
        "data_quality": {
            "samples": int(len(t)),
            "with_position": int(has_pos.sum()),
            "measured": int(valid_raw.sum()),
            "interpolated": int((has_pos & interp).sum()),
            "rejected": {r: int(c) for r, c in pd.Series(traj["reject_reason"]).value_counts().items() if r},
            "segments": int(len(np.unique(seg[seg >= 0]))),
        },
        "side_majority": int(np.round(np.average(side[has_pos], weights=np.maximum(w[has_pos], 1e-9))))
        if has_pos.any()
        else None,
    }
    derived = pd.DataFrame(
        {
            "t": t,
            "moving": moving,
            "zone": zones,
            "side": side,
            "lateral_m": lat,
            "depth_m": dep,
            "distance_from_base_m": d_base,
            "weight_s": w,
        }
    )
    return metrics, excursions_to_events(excursions) + events, derived


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    runs = []
    i = 0
    n = len(mask)
    while i < n:
        if mask[i]:
            j = i
            while j + 1 < n and mask[j + 1]:
                j += 1
            runs.append((i, j))
            i = j + 1
        else:
            i += 1
    return runs


def _excursions(t, x, y, d_base, speed, tang, seg, cfg: MovementConfig) -> list[dict]:
    """Movements away from base (> far threshold) and, if it happens, back (< return threshold)."""
    out = []
    n = len(t)
    i = 0
    while i < n:
        if not (np.isfinite(d_base[i]) and d_base[i] > cfg.far_from_base_m):
            i += 1
            continue
        # Walk back to where the player left the base region.
        s = i
        while s > 0 and seg[s - 1] == seg[i] and np.isfinite(d_base[s - 1]) and d_base[s - 1] > cfg.returned_to_base_m:
            s -= 1
        # Walk forward to the return (or the end of the segment).
        e = i
        while (
            e + 1 < n and seg[e + 1] == seg[i] and np.isfinite(d_base[e + 1]) and d_base[e + 1] > cfg.returned_to_base_m
        ):
            e += 1
        returned = e + 1 < n and seg[e + 1] == seg[i] and np.isfinite(d_base[e + 1])
        span = np.arange(s, e + 1)
        pk = s + int(np.nanargmax(d_base[span]))
        out_path = float(np.nansum(np.hypot(np.diff(x[s : pk + 1]), np.diff(y[s : pk + 1]))))
        straight = float(np.hypot(x[pk] - x[s], y[pk] - y[s]))
        directness = straight / out_path if out_path > 0.3 else None
        out.append(
            {
                "t_start": float(t[s]),
                "t_peak": float(t[pk]),
                "t_end": float(t[e + 1]) if returned else float(t[e]),
                "peak_distance_m": float(d_base[pk]),
                "peak_court_x": float(x[pk]),
                "peak_court_y": float(y[pk]),
                "outbound_path_m": out_path,
                "outbound_straight_m": straight,
                "directness": directness,
                "extra_distance_pct": (100 * (out_path - straight) / straight) if straight > 0.3 else None,
                "recovery_time_s": float(t[e + 1] - t[pk]) if returned else None,
                "returned": bool(returned),
                "peak_speed_mps": float(np.nanmax(speed[span])) if np.isfinite(speed[span]).any() else None,
                "smoothness_ldlj": _ldlj(np.nan_to_num(speed[span]), t[span]) if len(span) >= 7 else None,
            }
        )
        i = e + 1
    for e in out:
        if e["smoothness_ldlj"] is not None and not np.isfinite(e["smoothness_ldlj"]):
            e["smoothness_ldlj"] = None
    return out


def excursions_to_events(excursions: list[dict]) -> list[dict]:
    return [
        {
            "type": "excursion",
            "t": round(e["t_start"], 3),
            "t_end": round(e["t_end"], 3),
            **{k: (round(v, 3) if isinstance(v, float) else v) for k, v in e.items() if k not in ("t_start", "t_end")},
        }
        for e in excursions
    ]


def _mean(v: list) -> float | None:
    v = [a for a in v if a is not None and np.isfinite(a)]
    return float(np.mean(v)) if v else None


def _median(v: list) -> float | None:
    v = [a for a in v if a is not None and np.isfinite(a)]
    return float(np.median(v)) if v else None


COMPOSITE_FORMULA = (
    "score = 100 × Σ wₖ·sₖ / Σ wₖ over available components, where "
    "s_directness = mean outbound path directness (0–1); "
    "s_recovery = clip(1 − (median recovery time − 0.6 s) / 1.4 s, 0, 1); "
    "s_smoothness = clip((median LDLJ + 14) / 8, 0, 1); "
    "s_consistency = clip(1 − ready-position spread / 1.0 m, 0, 1)"
)


def _composite(fw: dict, cfg: MovementConfig) -> dict:
    """EXPERIMENTAL composite footwork index. Not a validated measure of efficiency."""
    comps: dict[str, float] = {}
    if fw["path_directness_mean"] is not None:
        comps["directness"] = float(np.clip(fw["path_directness_mean"], 0, 1))
    if fw["recovery_time_median_s"] is not None:
        comps["recovery"] = float(np.clip(1 - (fw["recovery_time_median_s"] - 0.6) / 1.4, 0, 1))
    if fw["smoothness_ldlj_median"] is not None:
        comps["smoothness"] = float(np.clip((fw["smoothness_ldlj_median"] + 14) / 8, 0, 1))
    if fw["positioning_consistency_m"] is not None:
        comps["consistency"] = float(np.clip(1 - fw["positioning_consistency_m"] / 1.0, 0, 1))
    weights = {k: v for k, v in cfg.composite_weights.items() if k in comps and v > 0}
    score = 100 * sum(weights[k] * comps[k] for k in weights) / sum(weights.values()) if weights else None
    return {
        "experimental": True,
        "score": score,
        "components": comps,
        "weights": cfg.composite_weights,
        "formula": COMPOSITE_FORMULA,
        "caveat": "Heuristic index for comparing one player's sessions under the same camera setup. "
        "Not validated against expert ratings; do not compare across players or setups.",
    }
