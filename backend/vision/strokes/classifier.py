"""Optional supervised stroke classifier (Layer B).

Pipeline: labelled stroke events (from the annotation UI) → fixed-length pose feature vectors →
one Random Forest per attribute (``shot_type``, ``stroke_side``) → evaluated with
session-grouped cross-validation (frames of one session never appear in both train and test).

A model is only used for inference after it has been trained on real labelled data and saved
with its model card. No model ships with the repository.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import joblib
import numpy as np

from vision.pose.landmarks import (
    L_ELBOW,
    L_SHOULDER,
    L_WRIST,
    NOSE,
    R_ELBOW,
    R_SHOULDER,
    R_WRIST,
    body_metrics,
)
from vision.strokes.detection import normalised_pose

FEATURE_NAMES = [
    "wrist_x_peak",
    "wrist_y_peak",
    "wrist_x_pre",
    "wrist_y_pre",
    "wrist_x_post",
    "wrist_y_post",
    "wrist_dy_pre_to_peak",
    "wrist_dy_peak_to_post",
    "wrist_dx_pre_to_peak",
    "wrist_dx_peak_to_post",
    "wrist_above_nose_peak",
    "elbow_angle_peak",
    "elbow_angle_min",
    "elbow_angle_max",
    "shoulder_angle_peak",
    "knee_flexion_max",
    "stance_width_max",
    "torso_orientation_peak",
    "wrist_lateral_vs_shoulder_axis",
    "peak_wrist_speed",
    "depth_from_net_m",
    "lateral_m",
    "player_speed_mps",
]

SHOT_TYPES = ["smash", "clear", "drop", "drive", "net_shot", "lift", "other"]
STROKE_SIDES = ["forehand", "backhand"]


def event_features(
    t: np.ndarray, xy: np.ndarray, vis: np.ndarray, t_event: float, hand: str = "right", context: dict | None = None
) -> np.ndarray:
    """Feature vector for one stroke event (NaN where unavailable; the model imputes)."""
    context = context or {}
    rel, _ = normalised_pose(xy, vis)
    m = body_metrics(xy, vis)
    p = int(np.clip(np.searchsorted(t, t_event), 0, len(t) - 1))
    pre = int(np.clip(np.searchsorted(t, t_event - 0.25), 0, len(t) - 1))
    post = int(np.clip(np.searchsorted(t, t_event + 0.2), 0, len(t) - 1))
    lo = int(np.clip(np.searchsorted(t, t_event - 0.4), 0, len(t) - 1))
    hi = int(np.clip(np.searchsorted(t, t_event + 0.3), 1, len(t)))
    wr, el = (R_WRIST, R_ELBOW) if hand == "right" else (L_WRIST, L_ELBOW)
    mirror = 1.0 if hand == "right" else -1.0
    w = rel[:, wr].copy()
    w[:, 0] *= mirror
    sh_vec = rel[p, R_SHOULDER] - rel[p, L_SHOULDER]
    sh_mid = (rel[p, R_SHOULDER] + rel[p, L_SHOULDER]) / 2
    lateral = float(np.dot(rel[p, wr] - sh_mid, sh_vec / (np.linalg.norm(sh_vec) + 1e-9))) * mirror
    speed = np.hypot(np.gradient(w[:, 0], t), np.gradient(w[:, 1], t))
    side = "left" if hand == "left" else "right"

    def nanmax(a):
        return float(np.nanmax(a)) if np.isfinite(a).any() else np.nan

    def nanmin(a):
        return float(np.nanmin(a)) if np.isfinite(a).any() else np.nan

    f = [
        w[p, 0],
        w[p, 1],
        w[pre, 0],
        w[pre, 1],
        w[post, 0],
        w[post, 1],
        w[p, 1] - w[pre, 1],
        w[post, 1] - w[p, 1],
        w[p, 0] - w[pre, 0],
        w[post, 0] - w[p, 0],
        rel[p, NOSE, 1] - rel[p, wr, 1],
        m[f"{side}_elbow_angle"][p],
        nanmin(m[f"{side}_elbow_angle"][lo:hi]),
        nanmax(m[f"{side}_elbow_angle"][lo:hi]),
        m[f"{side}_shoulder_angle"][p],
        nanmax(np.fmax(m["left_knee_flexion"][lo:hi], m["right_knee_flexion"][lo:hi])),
        nanmax(m["stance_width"][lo:hi]),
        m["torso_orientation"][p],
        lateral,
        nanmax(speed[lo:hi]),
        context.get("depth_from_net_m", np.nan),
        context.get("lateral_m", np.nan),
        context.get("player_speed_mps", np.nan),
    ]
    return np.asarray(f, float)


@dataclass
class TrainedStrokeModel:
    version: str
    target: str  # shot_type | stroke_side
    classes: list[str]
    pipeline: object
    card: dict

    def predict(self, X: np.ndarray, min_confidence: float = 0.6) -> list[dict]:
        proba = self.pipeline.predict_proba(X)
        out = []
        for row in proba:
            k = int(np.argmax(row))
            label = self.classes[k] if row[k] >= min_confidence else "uncertain"
            out.append(
                {
                    "label": label,
                    "confidence": float(row[k]),
                    "probabilities": {c: float(v) for c, v in zip(self.classes, row, strict=True)},
                }
            )
        return out

    def save(self, directory: Path) -> Path:
        directory.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {"pipeline": self.pipeline, "classes": self.classes, "target": self.target, "version": self.version},
            directory / "model.joblib",
        )
        (directory / "model_card.json").write_text(json.dumps(self.card, indent=2))
        return directory

    @classmethod
    def load(cls, directory: Path) -> TrainedStrokeModel:
        blob = joblib.load(directory / "model.joblib")
        card = json.loads((directory / "model_card.json").read_text())
        return cls(blob["version"], blob["target"], blob["classes"], blob["pipeline"], card)


def train_stroke_model(
    X: np.ndarray,
    y: list[str],
    groups: list[str],
    *,
    target: str,
    n_splits: int = 5,
    seed: int = 0,
    data_description: str = "",
) -> tuple[TrainedStrokeModel, dict]:
    """Train a Random Forest with session-grouped CV and return (model, evaluation report).

    The report contains out-of-fold macro-F1, per-class metrics and a confusion matrix. With
    fewer than two distinct groups no honest generalisation estimate is possible, so we refuse.
    """
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.impute import SimpleImputer
    from sklearn.metrics import classification_report, confusion_matrix, f1_score
    from sklearn.model_selection import GroupKFold
    from sklearn.pipeline import make_pipeline

    y_arr = np.asarray(y)
    g_arr = np.asarray(groups)
    n_groups = len(np.unique(g_arr))
    if n_groups < 2:
        raise ValueError("need labelled events from at least 2 sessions for grouped evaluation")
    classes = sorted(np.unique(y_arr).tolist())

    def make():
        return make_pipeline(
            SimpleImputer(strategy="median"),
            RandomForestClassifier(
                n_estimators=300, min_samples_leaf=2, class_weight="balanced", random_state=seed, n_jobs=-1
            ),
        )

    oof = np.empty(len(y_arr), dtype=object)
    oof_proba = np.zeros((len(y_arr), len(classes)))
    cv = GroupKFold(n_splits=min(n_splits, n_groups))
    for tr, te in cv.split(X, y_arr, g_arr):
        clf = make().fit(X[tr], y_arr[tr])
        pr = clf.predict_proba(X[te])
        cls_idx = [classes.index(c) for c in clf.classes_]
        full = np.zeros((len(te), len(classes)))
        full[:, cls_idx] = pr
        oof_proba[te] = full
        oof[te] = np.array(classes)[np.argmax(full, axis=1)]
    report = {
        "target": target,
        "n_events": int(len(y_arr)),
        "n_sessions": int(n_groups),
        "cv": f"GroupKFold(n_splits={min(n_splits, n_groups)}) grouped by session",
        "macro_f1": float(f1_score(y_arr, oof.astype(str), average="macro")),
        "per_class": classification_report(y_arr, oof.astype(str), output_dict=True, zero_division=0),
        "confusion_matrix": {
            "labels": classes,
            "matrix": confusion_matrix(y_arr, oof.astype(str), labels=classes).tolist(),
        },
        "expected_calibration_error": expected_calibration_error(
            oof_proba, np.array([classes.index(v) for v in y_arr])
        ),
        "class_counts": {c: int((y_arr == c).sum()) for c in classes},
    }
    final = make().fit(X, y_arr)
    version = f"{target}-rf-{datetime.now(UTC).strftime('%Y%m%d%H%M%S')}"
    card = {
        "model": version,
        "task": f"stroke {target} classification from pose features",
        "algorithm": "RandomForestClassifier(300 trees, balanced class weights) after median imputation",
        "features": FEATURE_NAMES,
        "classes": classes,
        "training_data": data_description,
        "evaluation": report,
        "intended_use": "Experimental coaching aid for footage similar to the training sessions.",
        "limitations": [
            "Pose-only features cannot see the shuttle; smash/clear/drop separation is inherently limited.",
            "Performance on new players, venues and camera angles is unknown unless they were in held-out sessions.",
        ],
        "created_at": datetime.now(UTC).isoformat(),
    }
    return TrainedStrokeModel(version, target, list(final.classes_), final, card), report


def expected_calibration_error(proba: np.ndarray, y_idx: np.ndarray, n_bins: int = 10) -> float | None:
    if len(proba) == 0:
        return None
    conf = proba.max(axis=1)
    pred = proba.argmax(axis=1)
    correct = (pred == y_idx).astype(float)
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(bins[:-1], bins[1:], strict=True):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            ece += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return float(ece)
