"""Read-side helpers over analysis artifacts (trajectory, heatmaps, playback annotations)."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from vision.analytics.heatmap import HeatmapConfig, compute_heatmap
from vision.pipeline import clean_json
from vision.trajectory.filtering import sample_weights


class NoResults(LookupError):
    pass


@lru_cache(maxsize=16)
def _load(results: str, mtime: float) -> dict:
    p = Path(results)
    data = {
        "positions": pd.read_parquet(p / "positions.parquet"),
        "tracks": pd.read_parquet(p / "tracks.parquet"),
        "summary": json.loads((p / "summary.json").read_text()),
        "events": json.loads((p / "events.json").read_text()),
    }
    pose = np.load(p / "pose.npz")
    data["pose_xy"] = pose["xy"]
    data["pose_vis"] = pose["visibility"]
    return data


def load_results(results: Path | None) -> dict:
    if results is None or not (results / "summary.json").exists():
        raise NoResults("this session has no completed analysis yet")
    return _load(str(results), (results / "summary.json").stat().st_mtime)


def trajectory_payload(res: dict, start: float | None, end: float | None, max_points: int = 5000) -> dict:
    pos = res["positions"]
    if "court_x" not in pos:
        return {"calibrated": False, "points": []}
    sel = pos
    if start is not None:
        sel = sel[sel["t"] >= start]
    if end is not None:
        sel = sel[sel["t"] <= end]
    step = max(1, int(np.ceil(len(sel) / max_points)))
    sel = sel.iloc[::step]
    cols = [
        "t",
        "court_x",
        "court_y",
        "court_x_raw",
        "court_y_raw",
        "speed",
        "interpolated",
        "segment",
        "position_source",
        "position_confidence",
        "reject_reason",
        "zone",
        "moving",
        "distance_from_base_m",
    ]
    cols = [c for c in cols if c in sel]
    records = sel[cols].replace({np.nan: None}).to_dict(orient="records")
    return clean_json({"calibrated": True, "step": step, "points": records})


def heatmap_payload(
    res: dict,
    *,
    start: float | None = None,
    end: float | None = None,
    zone: str | None = None,
    cell_m: float | None = None,
    sigma_m: float | None = None,
    normalize: str = "share",
    include_measured_only: bool = False,
) -> dict:
    pos = res["positions"]
    if "court_x" not in pos:
        return {"available": False, "reason": "no valid court calibration for this session"}
    t = pos["t"].to_numpy(float)
    x = pos["court_x"].to_numpy(float)
    y = pos["court_y"].to_numpy(float)
    seg = pos["segment"].to_numpy(int)
    valid = np.isfinite(x) & (seg >= 0)
    if include_measured_only:
        valid &= ~pos["interpolated"].to_numpy(bool)
    w = sample_weights(t, valid, seg)
    mask = np.ones(len(t), bool)
    if start is not None:
        mask &= t >= start
    if end is not None:
        mask &= t <= end
    if zone in ("front", "mid", "rear"):
        mask &= pos["zone"].astype(str).str.startswith(zone).to_numpy()
    cfg_d = res["summary"]["settings"]["heatmap"]
    cfg = HeatmapConfig(
        cell_m=cell_m or cfg_d["cell_m"],
        sigma_m=cfg_d["sigma_m"] if sigma_m is None else sigma_m,
        margin_m=cfg_d["margin_m"],
    )
    hm = compute_heatmap(x[mask], y[mask], w[mask], cfg)
    payload = hm.to_payload(normalize)
    payload.update(
        {
            "available": True,
            "start": start,
            "end": end,
            "zone": zone,
            "observed_s": float(w[mask].sum()),
            "synthetic": res["summary"]["synthetic"],
        }
    )
    return payload


def heatmap_object(res: dict, **kw):
    pos = res["positions"]
    t = pos["t"].to_numpy(float)
    x = pos["court_x"].to_numpy(float)
    y = pos["court_y"].to_numpy(float)
    seg = pos["segment"].to_numpy(int)
    valid = np.isfinite(x) & (seg >= 0)
    w = sample_weights(t, valid, seg)
    mask = np.ones(len(t), bool)
    if kw.get("start") is not None:
        mask &= t >= kw["start"]
    if kw.get("end") is not None:
        mask &= t <= kw["end"]
    cfg_d = res["summary"]["settings"]["heatmap"]
    cfg = HeatmapConfig(
        cell_m=cfg_d["cell_m"], sigma_m=kw.get("sigma_m", cfg_d["sigma_m"]) or 0.0, margin_m=cfg_d["margin_m"]
    )
    return compute_heatmap(x[mask], y[mask], w[mask], cfg), np.column_stack([x[mask], y[mask]])


def annotations_payload(res: dict, start: float, end: float) -> dict:
    """Per-sample overlay data for synchronized playback, for one time window.

    Coordinates are returned in *analysis-frame* pixels together with the scale factor, so the
    client maps them onto the displayed video regardless of its rendered size. Built from column
    arrays (not row by row) because the player requests a window every 30 s of playback.
    """
    pos = res["positions"]
    summary = res["summary"]
    t_all = pos["t"].to_numpy(float)
    idx = np.flatnonzero((t_all >= start) & (t_all <= end))
    tracks = res["tracks"]
    tr = tracks[(tracks["t"] >= start) & (tracks["t"] <= end)]
    people_by: dict[int, list] = {}
    if len(tr):
        boxes = np.round(tr[["x1", "y1", "x2", "y2"]].to_numpy(float), 1).tolist()
        scores = np.round(tr["score"].to_numpy(float), 2).tolist()
        for si, tid, b, sc in zip(tr["sample_index"].tolist(), tr["track_id"].tolist(), boxes, scores, strict=True):
            people_by.setdefault(int(si), []).append({"id": int(tid), "b": b, "s": sc})

    def col(name: str, digits: int) -> list:
        v = np.round(pos[name].to_numpy(float)[idx], digits)
        return [None if not np.isfinite(x) else float(x) for x in v]

    xy = res["pose_xy"][idx].astype(float)
    vis = res["pose_vis"][idx].astype(float)
    pose_ok = np.isfinite(xy).all(axis=(1, 2))
    pose_arr = np.round(np.concatenate([xy, vis[..., None]], axis=2), 2)
    sample = pos["sample_index"].to_numpy()[idx].tolist()
    tid = pos["track_id"].to_numpy(dtype=object)[idx]
    status = pos["status"].to_numpy(dtype=object)[idx].tolist()
    src = pos["position_source"].to_numpy(dtype=object)[idx].tolist()
    t_r, fx, fy, conf = col("t", 3), col("img_x", 1), col("img_y", 1), col("position_confidence", 2)
    has_court = "court_x" in pos
    if has_court:
        cx, cy, sp = col("court_x", 3), col("court_y", 3), col("speed", 2)
        interp = pos["interpolated"].fillna(False).to_numpy(bool)[idx].tolist()
    frames = []
    for k in range(len(idx)):
        f = {
            "t": t_r[k],
            "target": None if pd.isna(tid[k]) else int(tid[k]),
            "status": status[k],
            "people": people_by.get(int(sample[k]), []),
            "pose": pose_arr[k].tolist() if pose_ok[k] else None,
            "foot": None if fx[k] is None else [fx[k], fy[k]],
            "src": src[k],
            "conf": conf[k],
        }
        if has_court:
            f["court"] = None if cx[k] is None else [cx[k], cy[k]]
            f["speed"] = sp[k]
            f["interp"] = bool(interp[k])
        frames.append(f)
    return {
        "start": start,
        "end": end,
        "analysis_width": summary["analysis"]["width"],
        "analysis_height": summary["analysis"]["height"],
        "scale": summary["analysis"]["scale"],
        "frames": frames,
    }


def pose_series_payload(res: dict, landmark: str, window_s: float = 0.3) -> dict:
    """One landmark's image trajectory and the joint-angle proxies, raw next to smoothed.

    Smoothing is a Savitzky–Golay filter applied only inside runs of consecutive visible samples
    (never across gaps). It exists to *show* what temporal filtering does; the stored landmarks and
    the stroke detector keep using the raw values (the detector applies its own light smoothing).
    """
    from scipy.signal import savgol_filter

    from vision.pose.landmarks import IDX, MIN_VIS, body_metrics

    if landmark not in IDX:
        raise KeyError(f"unknown landmark '{landmark}'")
    k = IDX[landmark]
    t = res["positions"]["t"].to_numpy(float)
    xy = res["pose_xy"].astype(float)
    vis = res["pose_vis"].astype(float)
    fps = 1.0 / float(np.median(np.diff(t))) if len(t) > 1 else 15.0
    win = max(5, int(round(window_s * fps)) | 1)

    def smooth(v: np.ndarray) -> np.ndarray:
        out = np.full_like(v, np.nan)
        ok = np.isfinite(v)
        idx = np.flatnonzero(ok)
        if idx.size == 0:
            return out
        for run in np.split(idx, np.flatnonzero(np.diff(idx) > 1) + 1):
            if len(run) >= win:
                out[run] = savgol_filter(v[run], win, 2)
            else:
                out[run] = v[run]
        return out

    x = np.where(vis[:, k] >= MIN_VIS, xy[:, k, 0], np.nan)
    y = np.where(vis[:, k] >= MIN_VIS, xy[:, k, 1], np.nan)
    angles = body_metrics(xy, vis)
    series = {
        "x_raw": x,
        "x_smooth": smooth(x),
        "y_raw": y,
        "y_smooth": smooth(y),
        "visibility": vis[:, k],
    }
    for name in (
        "left_knee_flexion",
        "right_knee_flexion",
        "left_elbow_angle",
        "right_elbow_angle",
        "torso_lean_deg",
        "stance_width",
    ):
        series[f"{name}_raw"] = angles[name]
        series[f"{name}_smooth"] = smooth(angles[name])
    return clean_json(
        {
            "landmark": landmark,
            "window_samples": win,
            "t": t,
            "units": "x/y in analysis-frame pixels; angles in degrees "
            "(2-D monocular proxies); stance width in torso lengths",
            **series,
        }
    )
