"""Score one analysed session against a manual (or synthetic) annotation file.

Annotation format ``badmintonvision-annotations/1`` (all pixel coordinates in full-resolution
video pixels; see docs/annotation.md for the protocol)::

    {
      "format": "badmintonvision-annotations/1",
      "source": "manual" | "synthetic",
      "target_id": "P1",
      "frames": [{"t": 3.0, "people": [{"id": "P1", "bbox": [x1,y1,x2,y2], "ground_contact": [u,v] | null}]}],
      "calibration_points": [{"id": "near_baseline__left_doubles", "x": 102.5, "y": 655.0}],
      "court_positions": [{"t": 3.0, "court_xy": [2.1, 3.4]}],
      "strokes": [{"t": 4.12, "shot_type": "clear", "stroke_side": "forehand"}]
    }
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from vision.calibration.homography import Calibration, project
from vision.evaluation.metrics import (
    classification_metrics,
    detection_pr,
    event_detection_prf,
    mot_metrics,
    position_errors,
    target_tracking_accuracy,
)
from vision.geometry.court import keypoint_by_id

FORMAT = "badmintonvision-annotations/1"

# Coarse mapping used to score the heuristic shot attributes against fine-grained labels.
COARSE_SHOT = {
    "smash": "overhead",
    "clear": "overhead",
    "drop": "overhead",
    "overhead": "overhead",
    "overhead_unspecified": "overhead",
    "drive": "drive",
    "net_shot": "net",
    "net": "net",
    "lift": "lift",
    "underarm_unspecified": "underarm",
    "other": "other",
    "unknown": "unknown",
}


def truth_to_annotations(truth: dict, every_s: float = 0.5) -> dict:
    """Convert synthetic ground truth into the annotation format (clearly marked synthetic)."""
    frames = []
    next_t = 0.0
    for f in truth["frames"]:
        if f["t"] + 1e-6 < next_t:
            continue
        next_t = f["t"] + every_s
        frames.append(
            {
                "t": f["t"],
                "people": [
                    {"id": f"P{p['player']}", "bbox": p["bbox"], "ground_contact": None, "court_xy": p["court_xy"]}
                    for p in f["players"]
                ],
            }
        )
    return {
        "format": FORMAT,
        "source": "synthetic",
        "target_id": f"P{truth.get('target_player', 0)}",
        "frames": frames,
        "court_positions": [
            {"t": fr["t"], "court_xy": p["court_xy"]}
            for fr in frames
            for p in fr["people"]
            if p["id"] == f"P{truth.get('target_player', 0)}"
        ],
        "strokes": [
            {"t": s["t"], "shot_type": s["shot_type"], "stroke_side": s["stroke_side"]}
            for s in truth["strokes"]
            if s["player"] == truth.get("target_player", 0)
        ],
    }


def _nearest_sample(samples: pd.DataFrame, t: float, tol: float) -> int | None:
    ts = samples["t"].to_numpy()
    i = int(np.argmin(np.abs(ts - t)))
    return int(samples["sample_index"].iloc[i]) if abs(ts[i] - t) <= tol else None


def evaluate_session(
    results_dir: Path, annotations: dict, strokes: list[dict] | None = None, stroke_tolerance_s: float = 0.25
) -> dict:
    """Compute every metric the annotations support. Missing annotation types are skipped."""
    summary = json.loads((results_dir / "summary.json").read_text())
    samples = pd.read_parquet(results_dir / "samples.parquet")
    tracks = pd.read_parquet(results_dir / "tracks.parquet")
    dets = pd.read_parquet(results_dir / "detections.parquet")
    positions = pd.read_parquet(results_dir / "positions.parquet")
    scale = summary["analysis"]["scale"]
    fps = summary["analysis"]["fps"]
    tol = 0.5 / fps + 1e-3
    report: dict = {
        "source": annotations.get("source", "manual"),
        "synthetic": summary["synthetic"],
        "analysis": {
            k: summary["analysis"][k] for k in ("fps", "width", "height", "detector", "tracker", "performance_mode")
        },
    }

    frames = annotations.get("frames", [])
    if frames:
        gt_boxes, pred_boxes, gt_ids, pred_ids = {}, {}, {}, {}
        gt_target, pred_target = {}, {}
        pos_by_sample = positions.set_index("sample_index")
        target_id = annotations.get("target_id")
        for fr in frames:
            si = _nearest_sample(samples, fr["t"], tol)
            if si is None:
                continue
            gt_boxes[si] = [(p["id"], np.asarray(p["bbox"], float) * scale) for p in fr["people"]]
            d = dets[dets["sample_index"] == si]
            pred_boxes[si] = [
                (None, r[["x1", "y1", "x2", "y2"]].to_numpy(float), float(r["score"])) for _, r in d.iterrows()
            ]
            tr = tracks[tracks["sample_index"] == si]
            pred_ids[si] = [(int(r["track_id"]), r[["x1", "y1", "x2", "y2"]].to_numpy(float)) for _, r in tr.iterrows()]
            gt_ids[si] = gt_boxes[si]
            tgt = next((p for p in fr["people"] if p["id"] == target_id), None)
            if tgt is not None:
                gt_target[si] = np.asarray(tgt["bbox"], float) * scale
                if si in pos_by_sample.index:
                    row = pos_by_sample.loc[si]
                    box = np.array(
                        [row.get("x1", np.nan), row.get("y1", np.nan), row.get("x2", np.nan), row.get("y2", np.nan)],
                        float,
                    )
                    if not np.all(np.isfinite(box)):
                        tr_t = (
                            tracks[(tracks["sample_index"] == si) & (tracks["track_id"] == row["track_id"])]
                            if pd.notna(row["track_id"])
                            else tracks.iloc[:0]
                        )
                        box = tr_t[["x1", "y1", "x2", "y2"]].to_numpy(float)[0] if len(tr_t) else np.full(4, np.nan)
                    pred_target[si] = box
        report["detection"] = detection_pr(gt_boxes, pred_boxes)
        report["tracking"] = mot_metrics(gt_ids, pred_ids)
        report["tracking"]["note"] = (
            "computed on the annotated frames only; with sparse annotation an ID switch "
            "between two annotated frames counts once"
        )
        report["target"] = target_tracking_accuracy(gt_target, pred_target)
        report["annotated_frames"] = len(gt_boxes)

    cal = summary.get("calibration_used")
    if cal and annotations.get("calibration_points"):
        c = Calibration.from_dict(cal)
        ids = [p["id"] for p in annotations["calibration_points"]]
        img = np.array([[p["x"], p["y"]] for p in annotations["calibration_points"]], float)
        gt_court = np.array([[keypoint_by_id(i).x, keypoint_by_id(i).y] for i in ids])
        mapped = project(c.H, img)
        err = np.linalg.norm(mapped - gt_court, axis=1)
        report["calibration"] = {
            "n_points": len(ids),
            "mapped_error_mean_m": float(err.mean()),
            "mapped_error_max_m": float(err.max()),
            "fit_reprojection_error_mean_m": c.mean_error_m,
            "independent_points": [i for i in ids if i not in c.point_ids],
        }

    gt_pos = annotations.get("court_positions") or []
    if not gt_pos and cal and frames:
        # Derive court ground truth from annotated ground-contact clicks through the session's homography.
        c = Calibration.from_dict(cal)
        for fr in frames:
            tgt = next(
                (p for p in fr["people"] if p["id"] == annotations.get("target_id") and p.get("ground_contact")), None
            )
            if tgt:
                gt_pos.append(
                    {"t": fr["t"], "court_xy": project(c.H, np.array([tgt["ground_contact"]], float))[0].tolist()}
                )
    if gt_pos and "court_x" in positions:
        gt_xy, pred_xy, pred_raw = [], [], []
        for p in gt_pos:
            si = _nearest_sample(samples, p["t"], tol)
            if si is None:
                continue
            row = positions[positions["sample_index"] == si]
            gt_xy.append(p["court_xy"])
            pred_xy.append(row[["court_x", "court_y"]].to_numpy(float)[0] if len(row) else [np.nan, np.nan])
            pred_raw.append(row[["court_x_raw", "court_y_raw"]].to_numpy(float)[0] if len(row) else [np.nan, np.nan])
        report["position"] = {
            "filtered": position_errors(np.array(gt_xy), np.array(pred_xy)),
            "raw": position_errors(np.array(gt_xy), np.array(pred_raw)),
            "by_source": positions["position_source"].value_counts().to_dict(),
        }
        seg = positions["segment"].to_numpy()
        report["trajectory_continuity"] = {
            "segments": int(len(np.unique(seg[seg >= 0]))),
            "missing_rate": float(np.mean(~np.isfinite(positions["court_x"].to_numpy(float)))),
            "interpolated_rate": float(positions["interpolated"].fillna(False).astype(bool).mean()),
        }

    gt_strokes = annotations.get("strokes") or []
    if gt_strokes and strokes is not None:
        # Score the detector itself: every machine-detected event, regardless of later human review
        # (using review decisions here would leak the ground truth into the predictions).
        pred = [s for s in strokes if s.get("source") == "detected"]
        ev = event_detection_prf([s["t"] for s in gt_strokes], [s["t"] for s in pred], stroke_tolerance_s)
        matches = ev.pop("matches")
        gts = sorted(gt_strokes, key=lambda s: s["t"])
        prs = sorted(pred, key=lambda s: s["t"])
        side_true, side_pred, shot_true, shot_pred = [], [], [], []
        for i, j in matches:
            if gts[i].get("stroke_side"):
                side_true.append(gts[i]["stroke_side"])
                side_pred.append(prs[j].get("stroke_side") or "uncertain")
            if gts[i].get("shot_type"):
                shot_true.append(COARSE_SHOT.get(gts[i]["shot_type"], gts[i]["shot_type"]))
                shot_pred.append(
                    COARSE_SHOT.get(prs[j].get("shot_type") or "unknown", prs[j].get("shot_type") or "unknown")
                )
        report["strokes"] = {
            "events": ev,
            "stroke_side": classification_metrics(side_true, side_pred),
            "shot_type_coarse": classification_metrics(shot_true, shot_pred),
            "note": "shot types compared at coarse level (overhead / drive / net / lift) because the "
            "heuristics do not separate smash, clear and drop",
        }
    return report
