"""Evaluation metrics for each pipeline stage.

All functions take plain Python/numpy structures so they can score any annotation source:
the in-app annotation tool, MOTChallenge-style files, or synthetic ground truth.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np
from scipy.optimize import linear_sum_assignment

from vision.tracking.bytetrack import iou_matrix

Boxes = dict[int, list[tuple[object, np.ndarray]]]  # frame -> [(id, xyxy)]


def detection_pr(gt: Boxes, pred: dict[int, list[tuple[object, np.ndarray, float]]], iou_thr: float = 0.5) -> dict:
    """Precision / recall / F1 of person detection with one-to-one IoU matching per frame."""
    tp = fp = fn = 0
    for f in set(gt) | set(pred):
        g = np.array([b for _, b in gt.get(f, [])]).reshape(-1, 4)
        p = np.array([b for _, b, *_ in pred.get(f, [])]).reshape(-1, 4)
        if len(g) == 0:
            fp += len(p)
            continue
        if len(p) == 0:
            fn += len(g)
            continue
        iou = iou_matrix(g, p)
        r, c = linear_sum_assignment(-iou)
        m = int(np.sum(iou[r, c] >= iou_thr))
        tp += m
        fp += len(p) - m
        fn += len(g) - m
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    return {
        "precision": prec,
        "recall": rec,
        "f1": 2 * prec * rec / (prec + rec) if prec + rec else 0.0,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "iou_threshold": iou_thr,
    }


def mot_metrics(gt: Boxes, pred: Boxes, iou_thr: float = 0.5) -> dict:
    """CLEAR-MOT style MOTA / ID switches plus IDF1 (Ristani et al., 2016)."""
    frames = sorted(set(gt) | set(pred))
    last_match: dict[object, object] = {}
    id_switches = tp = fp = fn = 0
    gt_frames: dict[object, list[int]] = defaultdict(list)
    covered: dict[object, list[int]] = defaultdict(list)
    pair_overlap: dict[tuple[object, object], int] = defaultdict(int)
    gt_count: dict[object, int] = defaultdict(int)
    pred_count: dict[object, int] = defaultdict(int)
    for f in frames:
        g = gt.get(f, [])
        p = pred.get(f, [])
        for gid, _ in g:
            gt_count[gid] += 1
            gt_frames[gid].append(f)
        for pid, _ in p:
            pred_count[pid] += 1
        if not g or not p:
            fn += len(g)
            fp += len(p)
            continue
        iou = iou_matrix(np.array([b for _, b in g]), np.array([b for _, b in p]))
        # IDF1 counts every overlapping pair, independent of the per-frame assignment.
        for i, (gid, _) in enumerate(g):
            for j, (pid, _) in enumerate(p):
                if iou[i, j] >= iou_thr:
                    pair_overlap[(gid, pid)] += 1
        r, c = linear_sum_assignment(-iou)
        matched_g = set()
        matched_p = set()
        for i, j in zip(r, c, strict=True):
            if iou[i, j] < iou_thr:
                continue
            gid, pid = g[i][0], p[j][0]
            matched_g.add(i)
            matched_p.add(j)
            tp += 1
            covered[gid].append(f)
            if gid in last_match and last_match[gid] != pid:
                id_switches += 1
            last_match[gid] = pid
        fn += len(g) - len(matched_g)
        fp += len(p) - len(matched_p)
    n_gt = sum(gt_count.values())
    # IDF1: best one-to-one mapping of GT ids to predicted ids maximising identity TPs.
    gids = list(gt_count)
    pids = list(pred_count)
    idtp = 0
    if gids and pids:
        M = np.array([[pair_overlap.get((g_, p_), 0) for p_ in pids] for g_ in gids])
        r, c = linear_sum_assignment(-M)
        idtp = int(M[r, c].sum())
    idfn = n_gt - idtp
    idfp = sum(pred_count.values()) - idtp
    idf1 = 2 * idtp / (2 * idtp + idfp + idfn) if (2 * idtp + idfp + idfn) else 0.0
    # Continuity: number of fragments per GT track and its longest covered run.
    fragments, longest = [], []
    for gid, fs in gt_frames.items():
        cov = set(covered.get(gid, []))
        runs, best, cur = 0, 0, 0
        for f in fs:
            if f in cov:
                cur += 1
                if cur == 1:
                    runs += 1
                best = max(best, cur)
            else:
                cur = 0
        fragments.append(runs)
        longest.append(best / len(fs))
    return {
        "mota": 1 - (fn + fp + id_switches) / n_gt if n_gt else None,
        "idf1": idf1,
        "id_switches": id_switches,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "mean_fragments_per_track": float(np.mean(fragments)) if fragments else None,
        "mean_longest_run_share": float(np.mean(longest)) if longest else None,
        "iou_threshold": iou_thr,
    }


def target_tracking_accuracy(
    gt_target: dict[int, np.ndarray], pred_target: dict[int, np.ndarray], iou_thr: float = 0.5
) -> dict:
    """Share of annotated frames where the pipeline's chosen target overlaps the true target."""
    frames = sorted(gt_target)
    correct = wrong = missing = 0
    for f in frames:
        p = pred_target.get(f)
        if p is None or not np.all(np.isfinite(p)):
            missing += 1
        elif iou_matrix(gt_target[f][None], np.asarray(p)[None])[0, 0] >= iou_thr:
            correct += 1
        else:
            wrong += 1
    n = len(frames)
    return {
        "frames": n,
        "correct_share": correct / n if n else None,
        "wrong_person_share": wrong / n if n else None,
        "missing_share": missing / n if n else None,
    }


def position_errors(gt_xy: np.ndarray, pred_xy: np.ndarray) -> dict:
    gt_xy = np.asarray(gt_xy, float)
    pred_xy = np.asarray(pred_xy, float)
    ok = np.isfinite(gt_xy).all(axis=1) & np.isfinite(pred_xy).all(axis=1)
    e = np.linalg.norm(gt_xy[ok] - pred_xy[ok], axis=1)
    return {
        "n": int(ok.sum()),
        "missing_rate": float(1 - ok.mean()) if len(ok) else None,
        "mae_m": float(e.mean()) if e.size else None,
        "median_m": float(np.median(e)) if e.size else None,
        "p90_m": float(np.percentile(e, 90)) if e.size else None,
        "rmse_m": float(np.sqrt(np.mean(e**2))) if e.size else None,
        "mae_x_m": float(np.abs(gt_xy[ok, 0] - pred_xy[ok, 0]).mean()) if e.size else None,
        "mae_y_m": float(np.abs(gt_xy[ok, 1] - pred_xy[ok, 1]).mean()) if e.size else None,
    }


def event_detection_prf(gt_t: list[float], pred_t: list[float], tolerance_s: float = 0.25) -> dict:
    """Precision/recall/F1 of timestamped events with one-to-one matching within a tolerance."""
    gt_t = np.asarray(sorted(gt_t), float)
    pred_t = np.asarray(sorted(pred_t), float)
    matches: list[tuple[int, int]] = []
    if len(gt_t) and len(pred_t):
        cost = np.abs(gt_t[:, None] - pred_t[None, :])
        big = cost > tolerance_s
        cost = np.where(big, 1e6, cost)
        r, c = linear_sum_assignment(cost)
        matches = [(i, j) for i, j in zip(r, c, strict=True) if cost[i, j] < 1e5]
    tp = len(matches)
    fp = len(pred_t) - tp
    fn = len(gt_t) - tp
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    offsets = [float(pred_t[j] - gt_t[i]) for i, j in matches]
    return {
        "precision": prec,
        "recall": rec,
        "f1": 2 * prec * rec / (prec + rec) if prec + rec else 0.0,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tolerance_s": tolerance_s,
        "mean_abs_offset_s": float(np.mean(np.abs(offsets))) if offsets else None,
        "matches": matches,
    }


def classification_metrics(y_true: list[str], y_pred: list[str]) -> dict:
    from sklearn.metrics import confusion_matrix, f1_score

    labels = sorted(set(y_true) | set(y_pred))
    return {
        "n": len(y_true),
        "macro_f1": float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0))
        if y_true
        else None,
        "accuracy": float(np.mean(np.array(y_true) == np.array(y_pred))) if y_true else None,
        "confusion_matrix": {"labels": labels, "matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist()}
        if y_true
        else None,
    }
