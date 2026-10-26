"""Two-session comparison using normalised (duration- and frame-rate-independent) metrics."""

from __future__ import annotations


def _get(d: dict, path: str):
    cur = d
    for part in path.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return None
        cur = cur[part]
    return cur


# (key, label, path in metrics, unit, higher_is) — "higher_is" only drives neutral colouring.
COMPARABLE = [
    ("distance_per_min", "Distance per minute observed", "distance.per_minute_m", "m/min", "more"),
    ("mean_speed", "Mean speed", "speed.mean_mps", "m/s", "more"),
    ("moving_speed", "Mean speed while moving", "speed.mean_while_moving_mps", "m/s", "more"),
    ("p95_speed", "95th percentile speed", "speed.p95_mps", "m/s", "more"),
    ("moving_share", "Share of time moving", "speed.moving_share", "fraction", "more"),
    ("dir_changes_pm", "Direction changes per minute", "direction_changes_per_minute", "/min", "more"),
    ("zones_visited", "Singles zones visited", "zones.visited_pct", "%", "more"),
    ("front_share", "Front-court share", "zones.depth_share.front", "fraction", "neutral"),
    ("mid_share", "Mid-court share", "zones.depth_share.mid", "fraction", "neutral"),
    ("rear_share", "Rear-court share", "zones.depth_share.rear", "fraction", "neutral"),
    ("far_share", "Time > 1.5 m from base", "recovery.far_share", "fraction", "neutral"),
    ("recovery_median", "Median recovery time (proxy)", "recovery.recovery_time_median_s", "s", "less"),
    ("directness", "Path directness (proxy)", "footwork.path_directness_mean", "0–1", "more"),
    ("consistency", "Ready-position spread", "footwork.positioning_consistency_m", "m", "less"),
    ("area90", "Area holding 90% of time", "heatmap_area_90_m2", "m²", "neutral"),
    ("composite", "Experimental footwork index", "composite.score", "/100", "more"),
]

QUALITY = [
    ("coverage", "Target tracked (% of frames)", "quality.target_coverage_pct"),
    ("calibration_error", "Calibration mean error (m)", "quality.calibration.mean_error_m"),
    ("id_switches", "Possible identity switches", "quality.possible_id_switches"),
    ("gaps", "Tracking gaps ≥ 0.5 s", "quality.tracking_gaps"),
    ("analysis_fps", "Analysis frame rate", "analysis.fps"),
]


def compare(a_summary: dict, b_summary: dict) -> dict:
    rows = []
    for key, label, path, unit, better in COMPARABLE:
        va = _get(a_summary["metrics"], path)
        vb = _get(b_summary["metrics"], path)
        diff = (vb - va) if isinstance(va, (int, float)) and isinstance(vb, (int, float)) else None
        rel = (diff / abs(va) * 100) if diff is not None and va not in (0, None) else None
        rows.append(
            {
                "key": key,
                "label": label,
                "unit": unit,
                "a": va,
                "b": vb,
                "diff": diff,
                "rel_pct": rel,
                "direction": better,
            }
        )
    quality = []
    for key, label, path in QUALITY:
        quality.append({"key": key, "label": label, "a": _get(a_summary, path), "b": _get(b_summary, path)})
    warnings = []
    if a_summary.get("synthetic") != b_summary.get("synthetic"):
        warnings.append("one session is synthetic demo data and the other is not; the comparison is meaningless")
    if _get(a_summary, "analysis.performance_mode") != _get(b_summary, "analysis.performance_mode"):
        warnings.append(
            "sessions were analysed with different performance modes; speed and acceleration values "
            "depend on the sampling rate"
        )
    ca, cb = _get(a_summary, "quality.calibration.mean_error_m"), _get(b_summary, "quality.calibration.mean_error_m")
    if isinstance(ca, float) and isinstance(cb, float) and max(ca, cb) > 0.15:
        warnings.append("at least one calibration has > 15 cm mean error; position-based differences are uncertain")
    return {"metrics": rows, "quality": quality, "warnings": warnings}
