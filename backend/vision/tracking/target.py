"""Resolve the user's selected player into one identity across the whole video.

The tracker produces short-lived track ids: an id ends whenever the player is lost for longer
than the track buffer (occlusion by the net post, the other player, leaving the frame...).
This module chains those fragments into one *target* timeline:

1. **Seed** — the track that best overlaps the box the user clicked.
2. **Link** — when the current fragment ends, pick the fragment that starts within
   ``max_gap_s`` whose start is physically reachable (distance ≤ max speed × gap + slack) and
   whose shirt colour matches. Every link is recorded with its evidence for review.
3. **Corrections** — the user can override: "from time t the target is track k".
4. **Quality flags** — sudden appearance changes or impossible jumps inside one track id are
   flagged as possible identity switches; overlapping boxes mark partial occlusion.

Gaps are left as gaps. Nothing here invents a position for a frame without a detection.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from vision.calibration.homography import project
from vision.tracking.appearance import appearance_distance
from vision.tracking.bytetrack import iou_matrix

ASSUMED_PLAYER_HEIGHT_M = 1.75


@dataclass
class TargetConfig:
    max_gap_s: float = 3.0
    max_speed_mps: float = 7.0
    link_slack_m: float = 1.0
    max_link_appearance: float = 0.55
    switch_appearance_jump: float = 0.45
    switch_speed_mps: float = 10.0
    occlusion_iou: float = 0.15
    min_gap_event_s: float = 0.5


@dataclass
class TargetResult:
    frames: pd.DataFrame  # one row per sampled frame
    events: list[dict] = field(default_factory=list)
    seed_track: int | None = None


class TargetSelectionError(ValueError):
    pass


def _foot(df: pd.DataFrame) -> np.ndarray:
    return np.column_stack([(df["x1"].to_numpy() + df["x2"].to_numpy()) / 2, df["y2"].to_numpy()])


def _distance_m(p_img: np.ndarray, q_img: np.ndarray, h_px: float, H: np.ndarray | None) -> float:
    """Distance between two image foot points, in metres (via H) or approximated from box height."""
    if H is not None:
        a, b = project(H, np.vstack([p_img, q_img]))
        if np.all(np.isfinite(a)) and np.all(np.isfinite(b)):
            return float(np.linalg.norm(a - b))
    return float(np.linalg.norm(p_img - q_img) / max(h_px, 1.0) * ASSUMED_PLAYER_HEIGHT_M)


def select_seed(tracks: pd.DataFrame, t_sel: float, bbox_sel: np.ndarray, window_s: float = 0.5) -> int:
    near = tracks[(tracks["t"] - t_sel).abs() <= window_s]
    if near.empty:
        raise TargetSelectionError("no tracked people near the selected frame")
    # Prefer the sample closest in time, then highest IoU.
    best_t = near.loc[(near["t"] - t_sel).abs().idxmin(), "t"]
    cands = near[near["t"] == best_t]
    ious = iou_matrix(cands[["x1", "y1", "x2", "y2"]].to_numpy(), np.asarray(bbox_sel, float)[None])[:, 0]
    if ious.max() >= 0.3:
        return int(cands.iloc[int(np.argmax(ious))]["track_id"])
    # Fall back to the nearest box centre if the selection box is loose.
    c_sel = np.array([(bbox_sel[0] + bbox_sel[2]) / 2, (bbox_sel[1] + bbox_sel[3]) / 2])
    centres = np.column_stack([(cands["x1"] + cands["x2"]) / 2, (cands["y1"] + cands["y2"]) / 2])
    d = np.linalg.norm(centres - c_sel, axis=1)
    h = bbox_sel[3] - bbox_sel[1]
    if d.min() > 0.75 * h:
        raise TargetSelectionError("the selected box does not match any tracked person")
    return int(cands.iloc[int(np.argmin(d))]["track_id"])


def _summaries(tracks: pd.DataFrame, app: np.ndarray) -> dict[int, dict]:
    out = {}
    for tid, g in tracks.groupby("track_id"):
        g = g.sort_values("t")
        idx = g.index.to_numpy()
        k = min(5, len(g))
        out[int(tid)] = {
            "t0": float(g["t"].iloc[0]),
            "t1": float(g["t"].iloc[-1]),
            "foot0": _foot(g.iloc[:1])[0],
            "foot1": _foot(g.iloc[-1:])[0],
            "h0": float(g["y2"].iloc[0] - g["y1"].iloc[0]),
            "h1": float(g["y2"].iloc[-1] - g["y1"].iloc[-1]),
            "app0": app[idx[:k]].mean(axis=0),
            "app1": app[idx[-k:]].mean(axis=0),
            "n": len(g),
        }
    return out


def _chain(
    seed: int, summ: dict[int, dict], H, cfg: TargetConfig, forward: bool, events: list, used: set[int]
) -> list[int]:
    chain = [seed]
    cur = seed
    while True:
        s = summ[cur]
        best = None
        for tid, c in summ.items():
            if tid in used or tid == cur:
                continue
            gap = (c["t0"] - s["t1"]) if forward else (s["t0"] - c["t1"])
            if gap < -0.15 or gap > cfg.max_gap_s:
                continue
            p_from = s["foot1"] if forward else s["foot0"]
            p_to = c["foot0"] if forward else c["foot1"]
            h = s["h1"] if forward else s["h0"]
            dist = _distance_m(p_from, p_to, h, H)
            reach = cfg.max_speed_mps * max(gap, 0.0) + cfg.link_slack_m
            if dist > reach:
                continue
            app = appearance_distance(s["app1"] if forward else s["app0"], c["app0"] if forward else c["app1"])
            if app > cfg.max_link_appearance:
                continue
            cost = dist / reach + 1.5 * app
            if best is None or cost < best[0]:
                best = (cost, tid, gap, dist, app)
        if best is None:
            return chain
        cost, tid, gap, dist, app = best
        conf = float(np.clip(1.0 - 0.5 * cost, 0.05, 0.99))
        events.append(
            {
                "type": "identity_link",
                "t": round(s["t1"] if forward else s["t0"], 3),
                "from_track": cur,
                "to_track": tid,
                "gap_s": round(gap, 3),
                "distance_m": round(dist, 3),
                "appearance_distance": round(app, 3),
                "confidence": round(conf, 3),
            }
        )
        used.add(tid)
        chain.append(tid)
        cur = tid


def resolve_target(
    tracks: pd.DataFrame,
    appearance: np.ndarray,
    samples: pd.DataFrame,
    selection: dict,
    *,
    H: np.ndarray | None = None,
    corrections: list[dict] | None = None,
    config: TargetConfig | None = None,
) -> TargetResult:
    """Build the per-frame target timeline.

    ``tracks``: rows (sample_index, t, track_id, x1, y1, x2, y2, score), index aligned with
    ``appearance`` rows. ``samples``: every sampled frame (sample_index, frame_index, t).
    ``selection``: {"t": seconds, "bbox": [x1, y1, x2, y2]} in analysis-frame pixels.
    ``corrections``: [{"t": seconds, "track_id": int | None}] — None marks "target not visible".
    """
    cfg = config or TargetConfig()
    tracks = tracks.reset_index(drop=True)
    events: list[dict] = []
    samples = samples.sort_values("sample_index").reset_index(drop=True)
    out = samples[["sample_index", "frame_index", "t"]].copy()
    for col in ("track_id",):
        out[col] = pd.array([pd.NA] * len(out), dtype="Int64")
    for col in ("x1", "y1", "x2", "y2", "score", "max_overlap_iou"):
        out[col] = np.nan
    out["status"] = "missing"

    if tracks.empty:
        raise TargetSelectionError("no people were tracked in this video")
    summ = _summaries(tracks, appearance)
    seed = select_seed(tracks, float(selection["t"]), np.asarray(selection["bbox"], float))

    def build_assignment(seed_tid: int) -> dict[int, int]:
        used = {seed_tid}
        fwd = _chain(seed_tid, summ, H, cfg, True, events, used)
        bwd = _chain(seed_tid, summ, H, cfg, False, events, used)
        return {tid: i for i, tid in enumerate(bwd[::-1] + fwd[1:])}

    chain_ids = build_assignment(seed)
    # Per-sample assignment: track id owning the target at each time.
    assigned = pd.Series(pd.NA, index=out.index, dtype="Int64")
    by_track = {int(tid): g for tid, g in tracks.groupby("track_id")}
    for tid in chain_ids:
        g = by_track[tid]
        mask = out["sample_index"].isin(g["sample_index"])
        # When chain fragments overlap slightly, keep the earlier assignment.
        assigned[mask & assigned.isna()] = tid

    for corr in sorted(corrections or [], key=lambda c: c["t"]):
        t0 = float(corr["t"])
        later = [c["t"] for c in (corrections or []) if c["t"] > t0]
        t1 = min(later) if later else np.inf
        span = (out["t"] >= t0 - 1e-6) & (out["t"] < t1)
        assigned[span] = pd.NA
        tid = corr.get("track_id")
        events.append({"type": "manual_correction", "t": round(t0, 3), "to_track": tid})
        if tid is None:
            continue
        if int(tid) not in by_track:
            raise TargetSelectionError(f"track {tid} does not exist")
        local_used = {int(tid)}
        fwd = _chain(int(tid), summ, H, cfg, True, events, local_used)
        for ftid in fwd:
            g = by_track[ftid]
            m = span & out["sample_index"].isin(g["sample_index"]) & assigned.isna()
            assigned[m] = ftid

    out["track_id"] = assigned
    boxes = tracks[["sample_index", "track_id", "x1", "y1", "x2", "y2", "score"]].drop_duplicates(
        ["sample_index", "track_id"]
    )
    boxes = boxes.astype({"track_id": "Int64"})
    out = out.drop(columns=["x1", "y1", "x2", "y2", "score"]).merge(boxes, how="left", on=["sample_index", "track_id"])
    has = out["track_id"].notna().to_numpy()
    out.loc[has, "status"] = "tracked"

    # Overlap with other people (partial occlusion), vectorised per (target, other) pair.
    tgt = out.loc[has, ["sample_index", "track_id", "x1", "y1", "x2", "y2"]]
    pairs = tgt.merge(boxes, on="sample_index", suffixes=("", "_o"))
    pairs = pairs[pairs["track_id"] != pairs["track_id_o"]]
    out.loc[has, "max_overlap_iou"] = 0.0
    if not pairs.empty:
        ix1 = np.maximum(pairs["x1"], pairs["x1_o"])
        iy1 = np.maximum(pairs["y1"], pairs["y1_o"])
        ix2 = np.minimum(pairs["x2"], pairs["x2_o"])
        iy2 = np.minimum(pairs["y2"], pairs["y2_o"])
        inter = (ix2 - ix1).clip(lower=0) * (iy2 - iy1).clip(lower=0)
        area = (
            (pairs["x2"] - pairs["x1"]) * (pairs["y2"] - pairs["y1"])
            + (pairs["x2_o"] - pairs["x1_o"]) * (pairs["y2_o"] - pairs["y1_o"])
            - inter
        )
        pairs = pairs.assign(iou=inter / area.clip(lower=1e-9))
        mx = pairs.groupby("sample_index")["iou"].max()
        idx = out.index[has]
        out.loc[idx, "max_overlap_iou"] = out.loc[idx, "sample_index"].map(mx).fillna(0.0).to_numpy()
        out.loc[has & (out["max_overlap_iou"] >= cfg.occlusion_iou).to_numpy(), "status"] = "occluded"

    # Linked fragments (not the seed) are marked so the UI can show them.
    linked_mask = out["track_id"].notna() & (out["track_id"] != seed)
    out.loc[linked_mask & (out["status"] == "tracked"), "status"] = "linked"

    events.extend(_switch_flags(out, tracks, appearance, H, cfg))
    events.extend(_gap_events(out, cfg))
    events.sort(key=lambda e: e["t"])
    return TargetResult(frames=out, events=events, seed_track=seed)


def _switch_flags(out: pd.DataFrame, tracks: pd.DataFrame, app: np.ndarray, H, cfg: TargetConfig) -> list[dict]:
    """Flag frames where the same track id suddenly looks or moves like someone else."""
    events = []
    idx_lookup = {
        (int(s), int(t)): i for i, (s, t) in enumerate(zip(tracks["sample_index"], tracks["track_id"], strict=True))
    }
    valid = out[out["track_id"].notna()]
    for tid, g in valid.groupby("track_id"):
        if len(g) < 12:
            continue
        rows = np.array([idx_lookup[(int(s), int(tid))] for s in g["sample_index"]])
        A = app[rows]
        foot = _foot(g)
        t = g["t"].to_numpy()
        h = (g["y2"] - g["y1"]).to_numpy()
        for k in range(6, len(g) - 6):
            before = A[k - 6 : k].mean(axis=0)
            after = A[k : k + 6].mean(axis=0)
            d_app = appearance_distance(before, after)
            dt = max(t[k] - t[k - 1], 1e-3)
            dist = _distance_m(foot[k - 1], foot[k], h[k - 1], H)
            if d_app > cfg.switch_appearance_jump or dist / dt > cfg.switch_speed_mps:
                events.append(
                    {
                        "type": "possible_id_switch",
                        "t": round(float(t[k]), 3),
                        "track_id": int(tid),
                        "appearance_change": round(d_app, 3),
                        "implied_speed_mps": round(dist / dt, 2),
                    }
                )
    # Collapse bursts of flags into one event per second.
    merged: list[dict] = []
    for e in sorted(events, key=lambda e: e["t"]):
        if merged and e["t"] - merged[-1]["t"] < 1.0 and e["track_id"] == merged[-1]["track_id"]:
            continue
        merged.append(e)
    return merged


def _gap_events(out: pd.DataFrame, cfg: TargetConfig) -> list[dict]:
    events = []
    missing = out["track_id"].isna().to_numpy()
    t = out["t"].to_numpy()
    i = 0
    n = len(out)
    while i < n:
        if not missing[i]:
            i += 1
            continue
        j = i
        while j + 1 < n and missing[j + 1]:
            j += 1
        start = t[i - 1] if i > 0 else t[i]
        end = t[j + 1] if j + 1 < n else t[j]
        if end - start >= cfg.min_gap_event_s:
            events.append(
                {
                    "type": "tracking_gap",
                    "t": round(float(t[i]), 3),
                    "t_end": round(float(t[j]), 3),
                    "duration_s": round(float(t[j] - t[i]), 3),
                }
            )
        i = j + 1
    return events
